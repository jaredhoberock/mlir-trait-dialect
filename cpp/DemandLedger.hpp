// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::trait {

//===----------------------------------------------------------------------===//
// Vocabulary
//===----------------------------------------------------------------------===//
//
// A DEMAND is one monomorphic type -- a projection, or a claim spelled with
// one -- that some component asked to have resolved. A MISS is a demand a
// component declined to resolve: the ground-projection lookup found no unique
// impl to read, the unifier accepted an irreducible crossing without a binding,
// the resolver's projection engine failed and its caller swallowed that
// failure, obligation normalization had only one impl's own bindings to read,
// or a call claim withheld the license to consult module facts at all. This
// ledger records misses and nothing else; recording changes no return value
// anywhere.
//
// A miss is one component's decision and not the stage's answer. The two part
// company wherever the declining component writes nothing: obligation
// normalization hands its obligation straight on to impl resolution, and the
// unifier's acceptance leaves no trace at all, so in both cases whatever was
// not resolved is still on its way to a component that resolves it. DRAINABLE,
// below, is the predicate that keeps a component's decision apart from the
// stage's, and leavesDemandStanding is where the difference is written down.
//
// The STAGE POPULATION is the set of demands observed while the monomorphization
// stage owns the module -- the sub-phase that resolves impls and instantiates
// monomorphs, not the whole pass: the nominal materialization that runs after it
// hands the module on installs no sink. The ledger is reached through a
// thread-local sink the stage installs over its own span, so a demand raised
// outside that span finds no sink and contributes to the statistics alone. The
// ground-projection lookup also runs from op verifiers, which upstream parallel
// verification may schedule on worker threads; those calls name a verifier
// DemandOrigin, which is statistics-only, so the population stays
// single-threaded by construction.
//
// Three classes of mint sit outside the stage population by construction rather
// than by a list of exempt sites anyone has to maintain:
//   - what a verifier raises, which names a verifier DemandOrigin;
//   - what is minted after the stage hands the module on, where no sink is
//     installed;
//   - the spellings an impl generator stamps into the impl it is building,
//     which are its own construction rather than a question put to the module's
//     existing impls.
//
// DEDUPLICATION: a key is the demanded type itself, and types are uniqued, so
// one key stands for every observation of that demand. Flags belong to the
// OBSERVATION, and a key's flag state is the union over its observations. One
// key holds one observation's worth of PROVENANCE -- origin, parent, probe
// depth -- so the detail a key carries stays bounded however many times it is
// observed: the first observation's provenance stands until the first
// observation that was neither a probe nor a speculation and carried an
// enclosing frame arrives, which replaces it and is then kept.
//
// A key is DRAINABLE when at least one of its observations was unflagged AND
// came from an engine whose declining leaves the demand standing. One unflagged
// observation is a demand the stage really raised, whatever the other
// observations of that key were; the engine test is what keeps a demand the
// very next call serves out of the drain set. The drainable keys are what the
// stage reads back to decide what to resolve between its pattern drivers, so
// they are kept on every compile; the rest of what a ledger holds describes the
// population for a reader and is kept only when a census was asked for.
//
// KEYS ARE PER-SITE SPELLINGS. A recording site keys its observation by the type
// it is holding, and the sites hold different grades of the same demand: the
// lookup's callback receives a projection whose own arguments the replacer has
// not visited yet, while the resolver's engine holds one whose enclosing claim
// is already resolved. So one demand can stand under two keys, and a key count
// counts spellings. A reader that must join these keys against facts recorded
// elsewhere -- the proof memo is keyed by the resolved spelling -- normalizes
// them where it joins; the ledger does not, because normalizing at record time
// would put the lookup itself on the recording path of every site.
//
// A CANDIDATE PROBE is impl enumeration's per-candidate match attempt.
// Enumeration builds each candidate's self-claim substitution before any impl
// is selected, and that build re-enters the lookup; an observation recorded
// inside the re-entry is flagged PROBE-INTERNAL, because the lookup was asked
// about a candidate rather than about the demand under resolution.
//
// SPECULATIVE marks an observation raised while the resolver partitioned
// candidates by assumption satisfiability. Those sub-demands belong to
// candidates the partition may discard, so they are counted apart and a key
// observed only speculatively is never drainable.
//
// UNATTRIBUTED marks an observation recorded with no enclosing demand frame.
// Frames are pushed by this dialect's resolver entry points, its rewrite
// patterns and its declared-proof check; a demand raised by another dialect's
// in-stage pattern reaches the sink without one, and is recorded with no parent
// rather than dropped.
//
// Counts are per distinct demanded type per call and cover only demands that
// reached a component at all: the replacement callbacks that host the recording
// sites memoize within a call, and the resolver consults its negative memo
// before re-deriving a refuted application. The population is therefore a lower
// bound on the demand the stage raises.

//===----------------------------------------------------------------------===//
// Observation kinds
//===----------------------------------------------------------------------===//

/// Why the ground-projection lookup declined to resolve a monomorphic
/// projection. These are the lookup's exits that leave the projection spelled
/// as written, with the candidate-count exit split by which side of "exactly
/// one" it fell on.
enum class LookupMissReason : uint8_t {
  /// The projection's trait application names no trait in the module.
  TraitSymbolNotFound,
  /// No impl in the module binds the projection's trait application.
  NoCandidateImpl,
  /// Two or more impls bind it, so selection is a premise partition the
  /// read-only lookup does not perform.
  MultipleCandidateImpls,
  /// The unique impl has no usable binding for the projected associated type.
  AssociatedTypeBindingFailed,
  /// The unique impl's self claim does not specialize to the demanded claim.
  SelfClaimSubstitutionFailed,
};

/// The number of miss reasons, for the census partition.
inline constexpr unsigned numLookupMissReasons = 5;

/// The component whose decision an observation records.
enum class DemandEngine : uint8_t {
  /// resolveGroundProjectionsByLookup declined; carries a LookupMissReason.
  GroundProjectionLookup,
  /// The unifier accepted a monomorphic projection without resolving it.
  UnifierAcceptance,
  /// ImplResolver::resolveProjectionType failed and its caller kept the
  /// projection spelled as written.
  ResolverProjectionEngine,
  /// Obligation normalization left a monomorphic projection standing, having
  /// only one impl's own bindings to read.
  ObligationNormalization,
  /// A read of impl selection's recorded facts had none for the demand, and its
  /// caller left the demand spelled as written. The read cannot make selection
  /// run, so what it declines is for a step that can to serve.
  ///
  /// This engine is permanent. Its writers are the two patterns the instantiation
  /// driver keeps -- the one that resolves projections and the one that proves
  /// claim results -- and both are permanent for the same reason: the population
  /// they serve is the ground projections a substitution MINTS while the driver runs,
  /// which no sweep over what the module spells can reach, because the module
  /// never spells them. Resolving projections in the commit was measured against
  /// that population and refused: it cost 2-3% of a compile and still left that
  /// pattern applying 91 times, once per operation it rewrote.
  ReadOnlyResolver,
};

/// The number of engines, for the census partition.
inline constexpr unsigned numDemandEngines = 5;

/// Whether an unflagged observation by `engine` is, on its own, evidence that
/// the demand went unserved.
///
/// The test is whether the engine's declining puts the demand, still spelled as
/// written, into what the stage passes on. An engine that hands its caller a
/// type with the projection intact has left a demand for a later round to
/// serve; an engine that resolves nothing but also writes nothing has not,
/// because whatever it declined to resolve is still on its way to the
/// components that do resolve.
constexpr bool leavesDemandStanding(DemandEngine engine) {
  switch (engine) {
  // Each of these returns the demanded type spelled as written to a caller that
  // keeps that spelling: the lookup's replacement declines, and the resolver's
  // engine fails and its caller leaves the projection alone.
  case DemandEngine::GroundProjectionLookup:
  case DemandEngine::ResolverProjectionEngine:
  // A read of the recorded facts returns what it was asked about spelled as
  // written, and its caller declines the rewrite it was going to make. What it
  // declined is exactly what a step that may make selection run must serve.
  case DemandEngine::ReadOnlyResolver:
    return true;
  // The unifier writes nothing: it equated two spellings, or bound a variable,
  // and returned. What it accepted goes on to whatever the caller does next,
  // which routinely resolves it. So an observation here says the unifier did
  // not resolve the demand, not that the stage left it standing.
  case DemandEngine::UnifierAcceptance:
    return false;
  // Normalization reads one impl's own bindings and nothing else, and its
  // caller passes the obligation straight to impl resolution, whose first act
  // resolves the projections this engine left standing. So an observation here
  // says the normalization did not serve the demand, not that the stage did
  // not.
  case DemandEngine::ObligationNormalization:
    return false;
  }
  return false;
}

/// Bit per DemandEngine whose unflagged observation leaves the demand standing.
constexpr unsigned standingDemandEngineMask() {
  unsigned mask = 0;
  for (unsigned e = 0; e != numDemandEngines; ++e)
    if (leavesDemandStanding(static_cast<DemandEngine>(e)))
      mask |= 1u << e;
  return mask;
}

/// Whether one observation, on its own, makes its demand one a later round
/// could be asked to serve: raised for real by an engine whose declining leaves
/// it standing.
///
/// This is the drainability rule stated once. A key is drainable when any one
/// of its observations satisfies it, which is what `DemandRecord::isDrainable`
/// reads off the merged flags and what the drain below applies as each
/// observation arrives.
inline bool isDrainableObservation(DemandEngine engine, bool isReal) {
  return isReal && leavesDemandStanding(engine);
}

/// The component plus, for the lookup, which of its exits fired.
///
/// A miss reason names an exit of the ground-projection lookup and of no other
/// component, so a reason paired with any other engine has no meaning. The
/// named factories are the only way to build a pair; `get` builds one from
/// separately carried parts and refuses a pair that does not describe a single
/// component's decision.
class DemandObservationKind {
public:
  /// True when `reason` is present exactly for the lookup engine.
  static bool isWellFormed(DemandEngine engine,
                           std::optional<LookupMissReason> reason) {
    return reason.has_value() ==
           (engine == DemandEngine::GroundProjectionLookup);
  }

  /// The only constructor. It refuses a pair that does not describe one
  /// component's decision, so every kind that exists names one.
  static std::optional<DemandObservationKind>
  get(DemandEngine engine, std::optional<LookupMissReason> reason) {
    if (!isWellFormed(engine, reason))
      return std::nullopt;
    return DemandObservationKind(engine, reason);
  }

  // Each recording site knows its own engine, so these name a pair that is
  // well formed by construction.
  static DemandObservationKind lookupMiss(LookupMissReason reason) {
    return of(DemandEngine::GroundProjectionLookup, reason);
  }
  static DemandObservationKind unifierAcceptance() {
    return of(DemandEngine::UnifierAcceptance, std::nullopt);
  }
  static DemandObservationKind resolverProjectionEngine() {
    return of(DemandEngine::ResolverProjectionEngine, std::nullopt);
  }
  static DemandObservationKind obligationNormalization() {
    return of(DemandEngine::ObligationNormalization, std::nullopt);
  }
  static DemandObservationKind readOnlyResolver() {
    return of(DemandEngine::ReadOnlyResolver, std::nullopt);
  }

  DemandEngine getEngine() const { return engine; }
  std::optional<LookupMissReason> getMissReason() const { return reason; }

private:
  DemandObservationKind(DemandEngine engine,
                        std::optional<LookupMissReason> reason)
      : engine(engine), reason(reason) {}

  static DemandObservationKind of(DemandEngine engine,
                                  std::optional<LookupMissReason> reason) {
    auto kind = get(engine, reason);
    assert(kind && "a miss reason belongs to the lookup engine and to no other");
    return *kind;
  }

  DemandEngine engine;
  std::optional<LookupMissReason> reason;
};

/// Provenance an observation carries, orthogonal to its kind: any kind may be
/// observed speculatively, inside a candidate probe, both, or neither.
struct DemandFlags {
  bool speculative = false;
  bool probeInternal = false;
  bool unattributed = false;

  /// A demand the stage really raised: neither a candidate probe nor a
  /// discardable speculation. Attribution does not bear on this -- an
  /// unattributed observation is still a demand.
  bool isReal() const { return !speculative && !probeInternal; }
};

//===----------------------------------------------------------------------===//
// Call-site classification
//===----------------------------------------------------------------------===//

/// Which site raised a demand.
///
/// The sites that record carry a type and a module and say nothing about who
/// wants the answer, so every caller states it here. The parameter is not
/// defaulted anywhere it is threaded: a new caller must classify itself or the
/// build fails, which is what keeps this ledger's caller classification from
/// going stale.
enum class DemandOrigin : uint8_t {
  /// The module-capable replacer that stamps a specialized monomorph.
  MonomorphStampOut,
  /// The obligation recorder normalizing both sides before recording a proof.
  ProofRecording,
  /// The unifier reducing a ground projection a mid-solve binding minted.
  Unification,
  /// A call site specializing its callee at pass time.
  CallSiteSpecialization,
  /// A read of the recorded facts falling back to the module's impls for a
  /// projection impl selection has settled nothing for.
  RecordedFactRead,
  /// A call op's verifier comparing its formal and actual signatures.
  CallSignatureVerification,
  /// A proof op's verifier walking the proof structure it declares.
  ProofVerification,
  /// The unifier's module-free comparator, which a verifier reaches when it has
  /// no module to read facts from.
  ModuleFreeComparison,
};

/// Whether demand raised at this origin belongs to the stage population.
///
/// A verifier runs outside any stage and may run on a worker thread, so its
/// demand is counted by the statistics and never entered in the ledger.
inline bool recordsToLedger(DemandOrigin origin) {
  switch (origin) {
  case DemandOrigin::MonomorphStampOut:
  case DemandOrigin::ProofRecording:
  case DemandOrigin::Unification:
  case DemandOrigin::CallSiteSpecialization:
  case DemandOrigin::RecordedFactRead:
    return true;
  case DemandOrigin::CallSignatureVerification:
  case DemandOrigin::ProofVerification:
  case DemandOrigin::ModuleFreeComparison:
    return false;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// The demands a module spells
//===----------------------------------------------------------------------===//

/// Which subtrees one side of a demand walk passes over.
///
/// The two sides are chosen apart because the callers differ in what they are
/// asking. A reader tallying what the module spells anywhere wants a trait
/// header's projections; a caller asking which demands a round could still serve
/// does not, because a projection inside a template is resolved when the
/// template is cloned. A claim is a demand only where something is meant to
/// prove it, so the claim side is never gathered without at least the
/// infrastructure skip.
enum class DemandSkip : uint8_t {
  /// Nothing: every op the module holds, trait and impl headers included.
  Nothing,
  /// Trait, impl and proof ops and their whole subtrees. What they spell -- a
  /// trait's own requirements, an impl's assumptions, the projections in either
  /// header -- stands for good and is nothing to serve.
  Infrastructure,
  /// Those, and a still-polymorphic template function besides. What it spells is
  /// resolved when the template is cloned for a concrete instance, not by a
  /// round.
  InfrastructureAndTemplates,
};

/// The demands `module` spells: its monomorphic projections, and the unproven
/// monomorphic claims something is still meant to prove.
///
/// Result and block-argument types are what the stage's own leftover sweeps
/// walk, and an operand type is its producer's result type, so between them they
/// cover every type a later round would find to serve. `inAttributes` adds the
/// projections an op carries as attribute data instead, which no leftover sweep
/// reads and a pattern that rewrites an op's whole dictionary does.
///
/// The two sides carry their own skip because they answer different questions;
/// each caller states which it is asking.
/// The result is in the order the walk found them, so a caller putting these to
/// impl selection asks in an order one run repeats.
///
/// `origins`, when given, receives where each demand was first found spelled.
/// A caller that puts one to impl selection names that place while it does, so
/// what the ask raises underneath is attributed to the op carrying the spelling
/// rather than to the module.
llvm::SetVector<Type> demandsSpelledIn(ModuleOp module, bool inAttributes,
                                       DemandSkip projections,
                                       DemandSkip claims,
                                       DenseMap<Type, Location> *origins = nullptr);

//===----------------------------------------------------------------------===//
// Lowering a call
//===----------------------------------------------------------------------===//

/// Which step of lowering a call asked for proof bindings to be recorded.
///
/// Counting the steps apart is what says which of them a change stopped asking
/// at. These two are the whole of the asking a pass does: a call's parameter
/// specialization asks for a verifier alone, and a verifier's origin does not
/// record to a ledger.
///
/// XXX TODO: these columns exist to measure the re-derivation lowering a call
/// performs. They go when the call substitution's own resolution does.
enum class DerivationEntry : uint8_t {
  /// The call substitution's evidence walk, which asks once per type per
  /// fixed-point iteration of the factory that builds it.
  CallSubstitutionEvidence,
  /// The impl specialization a method call's callee is cloned under, verifying
  /// the self proof the call names.
  ImplSelfProof,
};

/// The number of steps, for the partition that reports them.
inline constexpr unsigned numDerivationEntries = 2;

/// One part of the work lowering a call does.
///
/// The parts nest: `Whole` spans a call's whole lowering and the three below
/// span steps inside it, so what those three leave unaccounted for is the
/// substitution application between them.
enum class CallLoweringPhase : uint8_t {
  /// Lowering one call, from the substitution it builds to the callee it names.
  Whole,
  /// Unifying the callee's formal signature with the call's actual one.
  Unification,
  /// Closing that substitution under projection and proof bindings.
  Closure,
  /// Getting or specializing the callee the closed substitution names.
  CalleeSpecialization,
};

/// The number of parts, for the decomposition that reports them.
inline constexpr unsigned numCallLoweringPhases = 4;

//===----------------------------------------------------------------------===//
// The ledger
//===----------------------------------------------------------------------===//

/// What the ledger knows about one demanded type once its observations are
/// merged. The counts are complete; the origin, parent and depth are one
/// observation's, promoted at most once as described under DEDUPLICATION above.
struct DemandRecord {
  DemandRecord(Location origin, Type parent, unsigned depth)
      : origin(origin), parent(parent), depth(depth) {}

  Location origin;
  Type parent;
  unsigned depth;
  /// Whether the provenance above came from an observation that was neither a
  /// probe nor a speculation and carried an enclosing frame. Until one such
  /// observation arrives the first observation's provenance stands.
  bool provenanceIsReal = false;
  unsigned observations = 0;
  /// Bit per DemandEngine value.
  unsigned engines = 0;
  /// Bit per DemandEngine that contributed an unflagged observation.
  unsigned realEngines = 0;
  /// Bit per LookupMissReason value.
  unsigned missReasons = 0;
  bool real = false;
  bool speculative = false;
  bool probeInternal = false;
  bool unattributed = false;

  /// Whether this key is a demand a later round could be asked to serve,
  /// derived rather than stored: an unflagged observation by an engine whose
  /// declining leaves the demand standing.
  bool isDrainable() const {
    return realEngines & standingDemandEngineMask();
  }
};

/// The stage's record of the demands its components declined to serve.
///
/// The ledger only appends. It is owned by the ImplResolver that spans the
/// stage and reached through the thread-local sink a DemandLedgerScope
/// installs, so a component with no resolver in scope can still record.
///
/// One ledger is written by one thread, because the sink is thread-local and
/// the stage runs on one thread. Should the stage itself come to run in
/// parallel, this holds only if each worker installs its own ledger and the
/// shards are merged at the end, or if recording moves behind the DemandOrigin
/// parameter, which every recording caller already carries.
class DemandLedger {
public:
  DemandLedger();

  /// Whether the census -- the per-key provenance, the flag cross-tabulation
  /// and the statistics -- is kept. Off unless the census environment variable
  /// is set. The drain below is not gated by this: a demand reaches it whenever
  /// a sink is installed, because the stage itself reads it.
  static bool isRecordingEnabled();

  /// Whether the ground-projection lookup's per-call postcondition and the
  /// stage-exit drainability cross-check are armed.
  static bool isPostconditionEnabled();

  /// Whether either switch is on. A site that counts what it cannot record --
  /// a demand raised while recording is suspended, or one raised by a verifier
  /// on an engine with no origin to read -- tests this rather than
  /// isDemandRecordingActive(), so its count survives the missing sink. With
  /// both switches off such a site does nothing at all.
  static bool areObservationsEnabled();

  /// Appends one observation of `demand`. The origin, the parent and the
  /// unattributed flag come from the frame stack; the caller supplies what only
  /// it knows.
  void record(Type demand, DemandObservationKind kind, DemandFlags flags,
              unsigned depth);

  /// The demanded types, in the order they were first observed.
  ArrayRef<Type> getKeys() const { return keys.getArrayRef(); }
  const DemandRecord *lookup(Type demand) const;

  /// The demands a later round could be asked to serve, in the order they were
  /// first raised.
  ///
  /// This is the one part of a ledger that is kept whatever the switches say,
  /// because the stage reads it to decide what to resolve between its pattern
  /// drivers. Everything else here -- the per-key provenance, the flag
  /// cross-tabulation, the statistics -- describes a population for a reader
  /// and is kept only when a census was asked for.
  ArrayRef<Type> getDrainableDemands() const { return drainable.getArrayRef(); }

  /// Bit per LookupMissReason on which the ground-projection lookup declined
  /// `demand` in an observation that made it drainable. Zero for a demand no
  /// lookup declined, and for one that is not drainable at all.
  unsigned getDrainableArms(Type demand) const;

  /// Sizes of the linear scans the resolver performs, so the indexing decision
  /// that would replace them can be made against measured work.
  ///
  /// XXX TODO: these counters size scans rather than describe the compiler.
  /// Delete them once the proofs and the impls a trait application selects
  /// among are both indexed rather than scanned.
  void countProofScan(size_t entries) {
    ++proofScans;
    proofScanEntries += entries;
  }
  void countProofCollisionScan(size_t entries) {
    ++proofCollisionScans;
    proofCollisionScanEntries += entries;
  }
  void countCandidateScan(size_t entries) {
    ++candidateScans;
    candidateScanEntries += entries;
  }

  /// What recursive proof verification did over this ledger's span, and how
  /// much of it repeated work already done.
  ///
  /// Verification is entered once per proven claim a type spells, and again for
  /// every obligation underneath it; the same obligation is reached from many
  /// call sites, each with its own evidence map born empty, so a repeat here is
  /// a derivation the map could not have skipped.
  void countProofVerification() { ++proofVerifications; }
  void countProofVerificationEarlyExit() { ++proofVerificationEarlyExits; }

  /// What the memo of completed derivations answered when it was asked, and
  /// how much of what was derived it can hold.
  ///
  /// A derivation is held only when this run computed the whole of its closure.
  /// One that exited early on a binding written before it began produces a
  /// closure nobody computed, so it is counted here and not held.
  void countProofDerivationMemoHit() { ++proofDerivationMemoHits; }
  void countProofDerivationMemoMiss() { ++proofDerivationMemoMisses; }
  void countProofDerivationRecorded() { ++proofDerivationsRecorded; }
  void countProofDerivationNotRecorded() { ++proofDerivationsNotRecorded; }

  /// What the record of per-application closures could answer where the memo of
  /// spelling pairs was asked, and how many nodes it described that the
  /// derivation running over them could not.
  ///
  /// The first pair is the record's coverage of the memo's own population,
  /// which is what a reader that must not derive at all is owed. The third is
  /// the closure a node exiting early on a binding written before this
  /// derivation began would otherwise have gone without.
  void countProofClosureAnswered() { ++proofClosuresAnswered; }
  void countProofClosureUnanswered() { ++proofClosuresUnanswered; }
  void countProofDerivationRecovered() { ++proofDerivationsRecovered; }

  /// Splits the asks the record could not answer by whether impl selection had
  /// recorded a proof for the application asked about.
  ///
  /// An ask the record cannot answer is a pair nothing has derived yet, and the
  /// two columns say whether anything could have: a proof this stage recorded
  /// was minted by an arm that could in principle have derived its closure
  /// then, and a proof it did not record came in with the module and reaches no
  /// arm at all. The two partition the unanswered asks a call site makes.
  void countFirstAskUnderRecordedProof() { ++firstAsksUnderRecordedProof; }
  void countFirstAskUnderUnrecordedProof() { ++firstAsksUnderUnrecordedProof; }

  /// Files one pair the record withdrew because two derivations of it reached
  /// two closures, so that the population it therefore answers for no longer is
  /// visible rather than silent.
  void countProofClosureWithdrawn() { ++proofClosuresWithdrawn; }

  /// Files one node the record answered for, which therefore derived nothing.
  ///
  /// A node is either replayed from the record or derived, so this and the
  /// derivations recorded below partition the nodes a derivation reaches once
  /// its spelling-pair memo has missed.
  void countProofClosureReplayed() { ++proofClosuresReplayed; }

  /// Files one derivation of a pair the record already held.
  ///
  /// A pair is derived once and read thereafter, so this is the work reading
  /// the record is meant to have stopped: what it counts is a reader that
  /// derived anyway, or an entry that stopped answering between the two asks.
  void countRecordedPairRederived() { ++recordedPairsRederived; }

  /// Files one evidence binding written into some substitution's map, and the
  /// high-water mark of the map that took it.
  ///
  /// A call site's evidence map holds the closure one derivation produced. A
  /// mark approaching the number of proofs the resolver has recorded says a
  /// substitution has stopped carrying its own closure and started carrying the
  /// whole record.
  void countEvidenceBinding(size_t bindingsAfter) {
    ++evidenceBindingsRecorded;
    evidenceBindingsMax = std::max(evidenceBindingsMax, bindingsAfter);
  }

  /// Files one visit that put a call to the read.
  void countCallLoweringVisit() { ++callLoweringVisits; }

  /// Files one callee specialization, saying whether the callee body had to be
  /// cloned or an instance already stood under the name the substitution
  /// mangles.
  void countCalleeSpecialization(bool cloned) {
    if (cloned)
      ++calleeClones;
    else
      ++calleeReuses;
  }

  /// Files one ask for the proof bindings a step of lowering a call needs.
  void countDerivationEntry(DerivationEntry entry) {
    ++derivationEntries[static_cast<unsigned>(entry)];
  }

  /// Files `nanoseconds` spent in one part of lowering a call.
  void countCallLoweringTime(CallLoweringPhase phase, uint64_t nanoseconds) {
    callLoweringNanoseconds[static_cast<unsigned>(phase)] += nanoseconds;
  }

  /// Pushes an enclosing demand, which observations recorded from here on carry
  /// as their parent. A site that knows the demand but not where it came from
  /// keeps the enclosing frame's origin.
  void pushFrame(Type demand);
  void pushFrame(Location origin);
  void popFrame();
  size_t getFrameDepth() const { return frames.size(); }

  /// Where the innermost open demand arose, or nothing when no frame is open.
  std::optional<Location> getInnermostFrameOrigin() const {
    if (frames.empty())
      return std::nullopt;
    return frames.back().origin;
  }

  /// Writes the census to standard error: one line per deduped key in
  /// first-observation order, then the partition of the population, then the
  /// statistics and the scan sizes.
  void dumpCensus() const;

  /// Writes how long each part of lowering a call took over this ledger's span,
  /// when the decomposition switch is set.
  ///
  /// Times differ from run to run, so they answer to their own switch and never
  /// join the census a baseline records.
  void reportCallLoweringProfile() const;

  /// Reports every drainable key `module` no longer has anything to serve.
  ///
  /// A drainable key is a demand a later round could be asked to serve, so when
  /// the stage finishes it must still be there to serve: the module still
  /// spells the demanded type, or an engine refused the demand outright and a
  /// later round would face the same refusal. A key that meets neither was
  /// served after it was recorded, which makes the drainability rule that
  /// admitted it an over-report. Like the lookup's postcondition this reports a
  /// gap in this ledger's own rules and never a fault in the program being
  /// compiled, so it goes to the census channel.
  ///
  /// Which keys the check reaches follows from the two exemptions. A key in
  /// `served` is one the engines that resolve it did resolve, so the stage
  /// served it on purpose and its spelling is gone by design; a key a round
  /// merely asked about is not exempt, because asking is not answering. Of the
  /// rest, a key carrying a miss arm is exempt because an engine that refused
  /// it outright and was never asked again would refuse it again. What remains
  /// -- a key with no arm that nothing served -- must still be spelled, or the
  /// drainability rule that admitted it over-reported.
  void reportServedDrainableKeys(
      ModuleOp module,
      const DenseSet<Type> &served = DenseSet<Type>()) const;

  /// Reports every drained key the stage neither served nor left standing, and
  /// fails when there is one.
  ///
  /// A round takes a key off the drain when nothing it could ask later would
  /// settle it differently: impl selection resolved it, or refused it on the
  /// arm no later resolution overturns and left its spelling for the stage's
  /// leftover walks to report. A key that is neither served nor still spelled
  /// was taken and dropped, and nothing downstream would say so -- which is why
  /// this is a failure of the stage rather than a line on the census channel.
  LogicalResult checkDrainedKeysSettled(ModuleOp module,
                                        const DenseSet<Type> &drained,
                                        const DenseSet<Type> &served) const;

  /// Reports every drainable demand still spelled at stage exit that no round
  /// served, and fails when there is one.
  ///
  /// checkDrainedKeysSettled covers the demands a round took off the drain and
  /// then lost; this covers the complementary case, a drainable demand no round
  /// settled -- one deferred to a round that never came, or one whose surviving
  /// spelling the stage's leftover-op walks do not reach because it lives on a
  /// block argument or in an attribute. Its population is the ledger's drainable
  /// keys rather than the drain's settled set, because a deferred demand never
  /// enters that set. A key still spelled here is one the stage undertook to
  /// serve and did not, which the leftover-op walks would have caught had it
  /// survived on an op result, so this runs after them as their backstop and a
  /// key it reaches is a failure of the stage.
  ///
  /// A demand several impls bind is refused for good and left spelled by design
  /// -- the module is rejected for the ambiguity by another check -- so a key
  /// carrying that arm is exempt, the reasoning reportServedDrainableKeys
  /// applies to a refused key, narrowed to the one arm no later resolution
  /// overturns.
  LogicalResult checkStandingDemandsServed(ModuleOp module,
                                           const DenseSet<Type> &served) const;

private:
  struct Frame {
    Type demand;
    Location origin;
  };

  /// What each reported statistic held when this ledger was built. A statistic
  /// counts a whole process and two censuses of one process are reachable, so
  /// the census reports the difference rather than the running total.
  struct StatisticBaseline {
    uint64_t residualToleranceAccepts = 0;
    uint64_t residualToleranceAcceptsGeneratorPending = 0;
    uint64_t residualToleranceAcceptsMultiCandidate = 0;
    uint64_t residualToleranceAcceptsHypothesis = 0;
    uint64_t residualToleranceAcceptsMixedOrOther = 0;
    uint64_t verifierLookupMisses = 0;
    uint64_t resolverProjectionMisses = 0;
    uint64_t unifierAcceptances = 0;
    uint64_t obligationNormalizations = 0;
    uint64_t obligationNormalizationMisses = 0;
    uint64_t withheldCallClaims = 0;
    uint64_t moduleFreeProjectionRejections = 0;
    uint64_t modulelessRegionProjections = 0;
  };

  llvm::SetVector<Type> keys;
  llvm::DenseMap<Type, DemandRecord> records;
  /// The drain: the keys of the census above that are drainable, kept apart
  /// from it because they are kept on every compile.
  llvm::SetVector<Type> drainable;
  llvm::DenseMap<Type, unsigned> drainableArms;
  SmallVector<Frame, 8> frames;
  uint64_t observationCount = 0;
  /// Observations cross-tabulated as they arrive, by engine and by the lookup's
  /// miss arm, against the three flag classes. A merged key no longer knows
  /// which of its observations carried which flag, so the cross-tabulation
  /// cannot be recovered from the records.
  uint64_t observationsByEngine[numDemandEngines][3] = {};
  uint64_t observationsByArm[numLookupMissReasons][3] = {};
  StatisticBaseline statisticsAtBirth;
  uint64_t proofScans = 0;
  uint64_t proofScanEntries = 0;
  uint64_t proofCollisionScans = 0;
  uint64_t proofCollisionScanEntries = 0;
  uint64_t candidateScans = 0;
  uint64_t candidateScanEntries = 0;
  uint64_t proofVerifications = 0;
  uint64_t proofVerificationEarlyExits = 0;
  uint64_t proofDerivationMemoHits = 0;
  uint64_t proofDerivationMemoMisses = 0;
  uint64_t proofDerivationsRecorded = 0;
  uint64_t proofDerivationsNotRecorded = 0;
  uint64_t proofClosuresAnswered = 0;
  uint64_t proofClosuresUnanswered = 0;
  uint64_t firstAsksUnderRecordedProof = 0;
  uint64_t firstAsksUnderUnrecordedProof = 0;
  uint64_t proofDerivationsRecovered = 0;
  uint64_t proofClosuresWithdrawn = 0;
  uint64_t proofClosuresReplayed = 0;
  uint64_t recordedPairsRederived = 0;
  uint64_t evidenceBindingsRecorded = 0;
  size_t evidenceBindingsMax = 0;
  uint64_t callLoweringVisits = 0;
  uint64_t calleeClones = 0;
  uint64_t calleeReuses = 0;
  uint64_t derivationEntries[numDerivationEntries] = {};
  uint64_t callLoweringNanoseconds[numCallLoweringPhases] = {};
};

//===----------------------------------------------------------------------===//
// Reaching the ledger
//===----------------------------------------------------------------------===//

/// True when a ledger is installed on this thread. A demand raised now reaches
/// its drain; whether anything beyond the drain is kept is what
/// `DemandLedger::areObservationsEnabled` decides.
bool isDemandSinkInstalled();

/// Where the innermost demand under resolution on this thread arose. Nothing
/// outside a stage span, and nothing inside one before a frame is opened.
std::optional<Location> currentDemandAnchor();

/// True when an observation recorded now would reach a ledger that keeps the
/// census. Every site that must walk or shape a type to know there was anything
/// to record tests this before doing that work.
bool isDemandRecordingActive();

/// Installs `ledger` as this thread's sink and restores the previous sink on
/// destruction, so nested spans compose.
///
/// The sink is installed whatever the switches say, because the drain is kept
/// on every compile. What the switches decide is how much of an observation the
/// ledger keeps once it arrives.
class DemandLedgerScope {
public:
  explicit DemandLedgerScope(DemandLedger &ledger);
  ~DemandLedgerScope();

  DemandLedgerScope(const DemandLedgerScope &) = delete;
  DemandLedgerScope &operator=(const DemandLedgerScope &) = delete;

private:
  DemandLedger *previous;
};

/// Suspends recording for its lifetime.
///
/// The stage verifies the whole module at points inside its own span, and a
/// verifier's comparisons reach the same recording sites the stage's do. The
/// sites that carry a DemandOrigin tell the two apart on their own; this
/// bracket covers the rest and is belt and braces for the ones that do. Demand
/// raised under it stays counted by the statistics, which is why the engines
/// the census alone could see count whenever observations are enabled rather
/// than only when a sink is installed.
///
/// No demand frame may span a suspension: a frame names an enclosing demand,
/// and suspending recording underneath it would leave that frame open across a
/// stretch the ledger knows nothing about. The constructor asserts it.
class DemandRecordingSuspension {
public:
  DemandRecordingSuspension();
  ~DemandRecordingSuspension();

  DemandRecordingSuspension(const DemandRecordingSuspension &) = delete;
  DemandRecordingSuspension &operator=(const DemandRecordingSuspension &) = delete;

private:
  DemandLedger *previous;
};

/// Marks work the compiler does only to compare one of its answers against
/// another, so that nothing here counts it.
///
/// A cross-check re-derives what an answer stood in for, or rebuilds what the
/// caller already holds. That is not work the program asked for: a ledger, a
/// statistic or a drain that counted it would report the checking and the
/// compilation together, and every population pinned off this channel would
/// move with whether a check was armed. Under this scope the sink is
/// uninstalled and every recording site is silent, so an armed check leaves the
/// census, the statistics and the drain saying what the compilation did.
///
/// Unlike a recording suspension this may open under a demand frame: a check
/// runs where the answer it checks was given, which is inside whatever the
/// stage was resolving. The frame is left where it is and goes on naming its
/// demand afterwards, because this replaces the sink rather than popping what
/// another sink holds; a frame opened underneath binds no sink and pushes
/// nothing.
class DemandCrossCheckScope {
public:
  DemandCrossCheckScope();
  ~DemandCrossCheckScope();

  DemandCrossCheckScope(const DemandCrossCheckScope &) = delete;
  DemandCrossCheckScope &operator=(const DemandCrossCheckScope &) = delete;

private:
  DemandLedger *previousLedger;
  bool previousChecking;
};

/// True while a cross-check is running on this thread, and its work is
/// therefore counted nowhere.
bool isCrossChecking();

/// Names what an enclosing site is working on, so observations recorded
/// underneath carry a parent, an origin, or both. A rewrite pattern knows where
/// the demand comes from but not which type will be demanded; the resolver's
/// entry points know the type but inherit the location of whoever called them.
/// Balanced by construction: a pattern that returns from the middle of its body
/// still pops, and it pops on the ledger it pushed to rather than on whatever
/// sink is installed when it goes out of scope.
class DemandFrame {
public:
  explicit DemandFrame(Type demand);
  explicit DemandFrame(Location origin);
  ~DemandFrame();

  DemandFrame(const DemandFrame &) = delete;
  DemandFrame &operator=(const DemandFrame &) = delete;

private:
  DemandLedger *ledger;
};

/// Counts re-entry into the ground-projection lookup's replacement callback.
///
/// Impl enumeration builds a candidate's self-claim substitution, which unifies,
/// which re-enters the lookup; this guard makes the re-entry visible so its
/// observations can be flagged probe-internal. It costs one thread-local
/// increment per projection the callback visits and nothing at all on a call
/// whose type carries no projection.
class LookupProbeScope {
public:
  LookupProbeScope();
  ~LookupProbeScope();

  LookupProbeScope(const LookupProbeScope &) = delete;
  LookupProbeScope &operator=(const LookupProbeScope &) = delete;

  /// How many lookup callbacks were already running when this one started.
  unsigned getEnclosingDepth() const { return enclosingDepth; }

private:
  unsigned enclosingDepth;
};

/// Marks the candidate partition, whose sub-demands belong to candidates the
/// partition may discard.
class SpeculationScope {
public:
  SpeculationScope();
  ~SpeculationScope();

  SpeculationScope(const SpeculationScope &) = delete;
  SpeculationScope &operator=(const SpeculationScope &) = delete;

private:
  bool previous;
};

//===----------------------------------------------------------------------===//
// Recording sites
//===----------------------------------------------------------------------===//
//
// Beside the ledger every site keeps an llvm::Statistic, and the sites differ
// in when theirs ticks, because they differ in what the tick costs. A site
// whose count is already in hand where the event happens ticks unconditionally,
// so -stats reports it with no census asked for. A site that must walk a type
// or test the shape of one to know there was an event -- the unifier's
// acceptance, obligation normalization's survivors, a detached region's
// stamped-in projections -- ticks only where observations are enabled, because
// paying for that walk on every compile is what the switches exist to avoid.
// Either way a demand raised where no sink is installed is counted rather than
// dropped.

/// Records a demand the ground-projection lookup declined, or counts it when
/// the origin is a verifier's.
void recordLookupMiss(Type demand, LookupMissReason reason, DemandOrigin origin,
                      unsigned enclosingDepth);

/// Records a monomorphic projection the unifier accepted without resolving, or
/// counts it when the origin is a verifier's.
void recordUnifierAcceptance(Type demand, DemandOrigin origin);

/// Records a projection the resolver's projection engine failed to resolve and
/// its caller left spelled as written.
void recordResolverProjectionMiss(Type demand);

/// Records a demand a read of impl selection's recorded facts had no answer
/// for, whose caller left it spelled as written.
void recordReadOnlyResolverMiss(Type demand);

/// Counts one demand a read of impl selection's recorded facts answered.
///
/// This is the positive twin of the miss above, and the two together say what
/// the recorded facts were able to carry: a read that only ever declined would
/// leave every demand for a later round.
void countReadOnlyResolverServe();

/// Records a monomorphic projection obligation normalization left standing, or
/// counts it when the origin is a verifier's.
void recordObligationNormalizationMiss(Type demand, DemandOrigin origin);

/// Counts one obligation normalized against an impl's own local bindings.
void countObligationNormalization();

/// Counts one call whose claim withheld the license to consult module facts.
void countWithheldCallClaim();

/// Counts a projection the unifier's module-free comparator rejected for
/// meeting a rigid type it could not be resolved against.
void countModuleFreeProjectionRejection();

/// Counts a region specialized with no module in scope that still carries a
/// monomorphic projection -- a mint no lookup could have served.
void countModulelessRegionProjection();

/// Counts the linear scans impl selection performs, through whatever ledger is
/// installed.
void countProofScan(size_t entries);
void countProofCollisionScan(size_t entries);
void countCandidateScan(size_t entries);

/// Counts what recursive proof verification did, through whatever ledger is
/// installed: one entry, one exit taken on a binding the caller's own evidence
/// map already held, what the memo of completed derivations answered, and how
/// much of what was derived it could hold.
///
/// These reach the ledger rather than a statistic because the memo is the
/// stage's and one thread's, so a population split across a process-wide
/// counter and a per-span one could not be read as one.
void countProofVerification();
void countProofVerificationEarlyExit();
void countProofDerivationMemoHit();
void countProofDerivationMemoMiss();
void countProofDerivationRecorded();
void countProofDerivationNotRecorded();

/// Counts what the record of per-application closures could answer where the
/// memo of spelling pairs was asked, and the nodes it described that the
/// derivation running over them could not.
void countProofClosureAnswered();
void countProofClosureUnanswered();
void countProofDerivationRecovered();

/// Splits the asks the record could not answer by whether impl selection had
/// recorded a proof for the application asked about.
void countFirstAskUnderRecordedProof();
void countFirstAskUnderUnrecordedProof();

/// Counts one pair the record of per-application closures withdrew because two
/// derivations of it reached two closures.
void countProofClosureWithdrawn();

/// Counts one node the record of per-application closures answered for.
void countProofClosureReplayed();

/// Counts one derivation of a pair the record already held.
void countRecordedPairRederived();

/// Counts one evidence binding written into a substitution's map, `bindingsAfter`
/// being the size of the map that took it.
void countEvidenceBinding(size_t bindingsAfter);

/// True when what the call-lowering instrument is told would be kept, so that a
/// site can skip the work of shaping what it would say and a span can skip
/// reading a clock.
bool isCallLoweringInstrumented();

/// Counts one visit that put a call to the read.
void countCallLoweringVisit();

/// Counts one callee specialization, `cloned` saying whether the callee body was
/// cloned rather than an existing instance returned.
void countCalleeSpecialization(bool cloned);

/// Counts one ask for the proof bindings a step of lowering a call needs.
void countDerivationEntry(DerivationEntry entry);

/// Times one part of lowering a call for as long as it is in scope.
///
/// The clock is read only where the instrument would keep the answer, so a
/// compilation nobody is measuring pays for one boolean per span.
class CallLoweringSpan {
public:
  explicit CallLoweringSpan(CallLoweringPhase phase);
  ~CallLoweringSpan();

  CallLoweringSpan(const CallLoweringSpan &) = delete;
  CallLoweringSpan &operator=(const CallLoweringSpan &) = delete;

private:
  CallLoweringPhase phase;
  bool timing;
  uint64_t startedAt;
};

//===----------------------------------------------------------------------===//
// The census channel
//===----------------------------------------------------------------------===//

/// Every census line starts with this word, so one run's census can be sifted
/// out of a whole compilation's standard error.
inline constexpr const char *demandCensusPrefix = "trait-demand-census";

/// The line a deduped key produces.
inline constexpr const char *demandCensusDemandPrefix =
    "trait-demand-census demand";

/// The lines the population's partition produces. An engine line's observation
/// count over its key count is that engine's revisit multiplier; the arm lines
/// refine the lookup engine's own line, so summing the two kinds double-counts
/// the lookup.
inline constexpr const char *demandCensusEnginePrefix =
    "trait-demand-census engine";
inline constexpr const char *demandCensusArmPrefix = "trait-demand-census arm";
inline constexpr const char *demandCensusSummaryPrefix =
    "trait-demand-census summary";

/// The lines the statistics produce, as deltas over this ledger's lifetime.
inline constexpr const char *demandCensusCounterPrefix =
    "trait-demand-census counter";

/// The line the resolver's scan sizes produce. It moves with how much work the
/// resolver does rather than with what the stage demanded, so a reader pinning
/// the demand population leaves it out.
inline constexpr const char *demandCensusScanPrefix = "trait-demand-census scan";

/// The line a monomorphic projection produces when it survives the lookup
/// inside the stage without any recording site having observed it. It reports a
/// gap in this ledger's wiring, never a fault in the program being compiled, so
/// it goes to the census channel and never to a diagnostic.
inline constexpr const char *demandCensusUnhookedPrefix =
    "trait-demand-census unhooked";

/// The line a drainable key produces when the stage went on to serve it. Like
/// the unhooked line it reports a gap in this ledger's rules and never a fault
/// in the program.
inline constexpr const char *demandCensusServedPrefix =
    "trait-demand-census served";

/// The line a type position produces when respelling it by proof-memo lookup
/// and respelling it by a substitution built from the whole memo disagree. Like
/// the two lines above it reports a gap in this dialect's own reasoning and
/// never a fault in the program being compiled.
inline constexpr const char *demandCensusRespellingDisagreementPrefix =
    "trait-demand-census respelling-disagreement";

/// The line a proof derivation produces when the closure held for it and the
/// closure deriving it again produces differ. Like the lines above it reports a
/// gap in this dialect's own reasoning and never a fault in the program.
inline constexpr const char *demandCensusProofDerivationDisagreementPrefix =
    "trait-demand-census proof-derivation-disagreement";

/// Setting this makes the stage record demands and write the census at its end.
///
/// Both switches are read once, at library load, so a caller that sets either
/// variable inside its own process afterwards gets nothing. Every in-tree
/// reader runs one compilation per process, which is what makes reading them
/// once the right trade.
inline constexpr const char *demandCensusEnvironmentVariable =
    "TRAIT_DEMAND_CENSUS";

/// Setting this arms the lookup's per-call postcondition and the stage-exit
/// drainability cross-check.
inline constexpr const char *demandCensusCheckEnvironmentVariable =
    "TRAIT_DEMAND_CENSUS_CHECK";

/// Writes one unhooked-mint line for `demand`.
void reportUnhookedMint(Type demand);

//===----------------------------------------------------------------------===//
// The call-lowering decomposition channel
//===----------------------------------------------------------------------===//
//
// How long each part of lowering a call takes is a measurement rather than a
// fact about the program, so it answers to a switch of its own and writes on a
// marker of its own. A baseline records what a compilation did; a time differs
// between two runs of the same compilation, so nothing that reads a baseline
// reads this.

/// The line the decomposition produces.
inline constexpr const char *callLoweringProfilePrefix =
    "trait-call-lowering-profile";

/// Setting this writes the decomposition of the work lowering trait calls does.
/// It is read once, at library load, like the census switches.
inline constexpr const char *callLoweringProfileEnvironmentVariable =
    "TRAIT_CALL_LOWERING_PROFILE";

//===----------------------------------------------------------------------===//
// The collector-walk channel
//===----------------------------------------------------------------------===//
//
// A round puts to impl selection what the ledger's engines declined. A walk over
// the module finds what those engines meet -- every demand the module spells
// where a pattern would have met it -- and the two populations must be compared
// before one can stand in for the other. The comparison is a measurement of the
// compiler rather than a fact about the program, so it answers to a switch of
// its own and writes on a marker of its own.

/// The line one round's comparison produces.
inline constexpr const char *collectorWalkPrefix = "trait-collector-walk";

/// Setting this writes that comparison, one line per round. It is read once, at
/// library load, like the census switches.
inline constexpr const char *collectorWalkEnvironmentVariable =
    "TRAIT_COLLECTOR_WALK";

/// True when the comparison would be written, so that the walk it needs is taken
/// only where someone is reading it.
bool isCollectorWalkReported();

//===----------------------------------------------------------------------===//
// The stage-record channel
//===----------------------------------------------------------------------===//
//
// Beside the census of what the stage declined to serve, the stage reports what
// it did: the facts impl resolution recorded, the rewrite events each run of a
// greedy pattern driver raised, and how much of the module each claim-respelling
// sweep touched. Rescheduling the stage's work must leave the first alone and
// must not grow the other two, so all three are recorded per run and read
// against a baseline.
//
// These lines share the census switch, because a reader that wants either wants
// both, and carry their own marker, because a reader pinning the demand
// population must not have to re-record it whenever a work count moves.

/// Every stage-record line starts with this word.
inline constexpr const char *stageRecordPrefix = "trait-stage-record";

/// The line one recorded fact of impl resolution produces. These lines name
/// trait applications, and a monomorphic type's name embeds a mangled hash, so
/// like the census's per-key lines they are read where a module is small rather
/// than recorded across a corpus.
inline constexpr const char *stageRecordFactPrefix = "trait-stage-record fact";

/// The line the recorded facts produce together: a digest over the canonical
/// rendering of all of them, and the counts behind it.
inline constexpr const char *stageRecordDigestPrefix =
    "trait-stage-record digest";

/// The line one run of a greedy pattern driver produces.
inline constexpr const char *stageRecordRewritesPrefix =
    "trait-stage-record rewrites";

/// The line one round of the stage produces. It names what the round did and
/// digests the facts impl resolution had recorded when the round ended, so a
/// reader can see which round settled what.
inline constexpr const char *stageRecordRoundPrefix = "trait-stage-record round";

/// The line one claim-respelling sweep produces.
inline constexpr const char *stageRecordRespellingPrefix =
    "trait-stage-record respelling";

} // end mlir::trait
