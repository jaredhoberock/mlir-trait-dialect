// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <memory>

namespace mlir::trait {

// Interface to generate a new ImplOp for the wanted claim, or fail
struct ImplGenerator {
  virtual ~ImplGenerator() = default;

  // Creates exactly one new ImplOp for the wanted claim, or fails.
  // Upon success, returns the newly created ImplOp whose self claim
  // must unify with wanted.
  //
  // A generator only builds IR, so an OpBuilder suffices. The caller places
  // the builder where a generated impl belongs before calling: a generator
  // that creates its impl at the ambient insertion point puts it wherever the
  // caller left the builder. On the way out a generator may leave the
  // insertion point parked inside the IR it just generated, and restoring it
  // is again the caller's responsibility.
  //
  // A generated impl is IR nothing revisits unless the caller hears about it,
  // so the builder must carry a listener and the caller must act on what it
  // hears. When generation runs underneath a greedy pattern driver, that means
  // passing the driver's active PatternRewriter itself rather than a builder of
  // its own: the driver's listener is what places generated ops on its
  // worklist, and a builder without it silently changes which ops the driver
  // revisits. A caller with no driver running must instead scan what it
  // generated itself, which is what a caller that counts insertions and then
  // runs a driver over the whole module does.
  virtual FailureOr<ImplOp>
  generateImpl(TraitOp trait,
               ClaimType wanted,
               OpBuilder &builder) const = 0;
};

// Composite that itself behaves like an ImplGenerator
class ImplGeneratorSet : public ImplGenerator {
  public:
    inline FailureOr<ImplOp>
    generateImpl(TraitOp trait,
                 ClaimType wanted,
                 OpBuilder &builder) const override {
      // return the first successful result of all generators
      for (const auto &g : generators) {
        // a generator may park the insertion point in the IR it generated,
        // so each attempt starts from the insertion point we were handed
        OpBuilder::InsertionGuard guard(builder);
        auto maybeImpl = g->generateImpl(trait, wanted, builder);
        if (succeeded(maybeImpl))
          return maybeImpl;
      }
      return failure();
    }

    inline ImplGeneratorSet &add(std::unique_ptr<ImplGenerator> g) {
      generators.emplace_back(std::move(g));
      return *this;
    }

    template<typename... Ts>
    ImplGeneratorSet &add() {
      (add(std::make_unique<Ts>()), ...);
      return *this;
    }

  private:
    SmallVector<std::unique_ptr<ImplGenerator>,4> generators;
};

/// Why impl selection refused a trait application.
///
/// Selection wants exactly one candidate whose assumptions hold, and the two
/// ways to miss that differ in whether the answer can still change: a generator
/// can supply the impl that is missing, whereas a second satisfiable candidate
/// can only ever be joined by more.
enum class RefutationArm : uint8_t {
  /// No candidate's assumptions were satisfiable and generation supplied none.
  NoSatisfiableCandidate,
  /// Two or more candidates' assumptions were satisfiable, so the application
  /// is proven by no unique impl.
  MultipleSatisfiableCandidates,
};

/// What impl selection settled on for one trait application: the impl it chose,
/// or the arm on which it refused.
///
/// A selection that carried both would name an impl it had refused to select,
/// and one that carried neither would say nothing at all. The only constructor
/// refuses either, so every outcome that exists names one of the two.
class ResolutionOutcome {
public:
  /// True when exactly one of an impl and a refutation arm is present.
  static bool isWellFormed(ImplOp impl, std::optional<RefutationArm> arm) {
    return static_cast<bool>(impl) != arm.has_value();
  }

  /// The only constructor. It refuses a pair that is not one outcome.
  static std::optional<ResolutionOutcome> get(ImplOp impl,
                                              std::optional<RefutationArm> arm) {
    if (!isWellFormed(impl, arm))
      return std::nullopt;
    return ResolutionOutcome(impl, arm);
  }

  // Each recording site knows which outcome it reached, so these name a pair
  // that is well formed by construction.
  static ResolutionOutcome selected(ImplOp impl) { return of(impl, std::nullopt); }
  static ResolutionOutcome refused(RefutationArm arm) {
    return of(ImplOp(), arm);
  }

  bool isRefusal() const { return arm.has_value(); }

  ImplOp getImpl() const {
    assert(!isRefusal() && "a refusal names no impl");
    return impl;
  }

  RefutationArm getRefutationArm() const {
    assert(isRefusal() && "a selection was refused on no arm");
    return *arm;
  }

private:
  ResolutionOutcome(ImplOp impl, std::optional<RefutationArm> arm)
      : impl(impl), arm(arm) {}

  static ResolutionOutcome of(ImplOp impl, std::optional<RefutationArm> arm) {
    auto outcome = get(impl, arm);
    assert(outcome && "an outcome is either a selected impl or a refusal");
    return *outcome;
  }

  ImplOp impl;
  std::optional<RefutationArm> arm;
};

// Memoization state for pure impl resolution (no IR mutations).
struct ResolutionMemo {
  // Maps a fully-concrete trait application to the impl selected for it, or to
  // the arm on which selection was refused when no unique impl exists.
  DenseMap<TraitApplicationAttr, ResolutionOutcome> chosen;

  // The applications impl selection is part-way through, outermost first. A
  // repeat is a resolution cycle; the count of frames naming one trait is how
  // far the obligation chain has recursed through that trait, which is what
  // bounds a chain whose every step is a new application.
  SmallVector<TraitApplicationAttr> visiting;

  // A memo for assumptionsSatisfiableFor
  // For every (ImplOp, TraitApplicationAttr) in this set, the ImplOp's
  // assumptions are known to be satisfiable for the given TraitApplicationAttr
  // We only memoize satisfiable results because new proofs appear in the IR
  // as resolution unfolds
  DenseSet<std::pair<ImplOp,TraitApplicationAttr>> assumptionsKnownSatisfiable;
};

/// The impl selected for a claim, paired with the normalized claim used for
/// selection.
///
/// Projection normalization is part of impl resolution. Callers that specialize
/// the selected impl must use this claim, not the original source spelling, so
/// selection and substitution agree on the same semantic type arguments.
struct ResolvedImpl {
  ImplOp impl;
  ClaimType selectedClaim;
};

/// How many frames from each end of a chain a refusal names. A chain at the
/// depth limit is a hundred-odd frames of the same shape; its ends say where it
/// started and what it grew into, and the frames between them say nothing more.
constexpr size_t kChainEndsNamed = 3;

/// Attaches the ends of `chain` to `diagnostic`, one note per frame through
/// `name`, with a note standing in for the frames between them.
template <typename FrameT>
void nameChainEnds(InFlightDiagnostic &diagnostic, ArrayRef<FrameT> chain,
                   llvm::function_ref<void(InFlightDiagnostic &, FrameT)> name) {
  if (chain.size() <= 2 * kChainEndsNamed) {
    for (const FrameT &frame : chain)
      name(diagnostic, frame);
    return;
  }
  for (const FrameT &frame : chain.take_front(kChainEndsNamed))
    name(diagnostic, frame);
  diagnostic.attachNote() << "... " << chain.size() - 2 * kChainEndsNamed
                          << " more frame(s) elided";
  for (const FrameT &frame : chain.take_back(kChainEndsNamed))
    name(diagnostic, frame);
}

/// The template instantiations one stage run has cut.
///
/// Each instance is cut for a template at a call standing inside another
/// function, so the instances form a forest and what matters about one is how
/// many instances of the SAME template stand on the path that reaches it. A
/// template that instantiates itself at a larger type mints a distinct instance
/// at every step, so no cycle guard sees a repeat; that count is what tells
/// such a chain from a deep but finite nest of distinct templates.
///
/// An instance is named by the function it was cut into. Nothing erases a
/// function while the stage runs, so a recorded name stands for as long as the
/// chain does.
class InstantiationChain {
public:
  /// How many instances of `templateKey` stand on the chain reaching
  /// `instance`, counting `instance` itself. A function this has never
  /// recorded is a root and stands at zero.
  unsigned depthAt(Operation *instance, Attribute templateKey) const;

  /// Records that `instance` was cut for `templateKey` at a call inside
  /// `parent`, and answers the depth it now stands at.
  ///
  /// An instance already recorded keeps the chain it was first cut on, and an
  /// instance that is its own parent -- a call whose specialization reached the
  /// very function it stands in -- records nothing, so the chain stays a
  /// forest.
  unsigned note(Operation *instance, Operation *parent, Attribute templateKey);

  /// The frames from the root down to `instance`, each a function and the
  /// template it was cut for. This is what a refusal at the depth limit names.
  SmallVector<std::pair<Operation *, Attribute>>
  chainTo(Operation *instance) const;

  /// The deepest per-template count any instance reached, which is the
  /// quantity the depth limit stands over.
  unsigned getMaxDepth() const { return maxDepth; }

  /// Says a call refused to instantiate because the chain reached the limit.
  void noteLimitReached() { limitReached = true; }

  /// Whether any call refused on the depth limit. A greedy driver treats that
  /// refusal as a pattern that did not apply, so the stage reads this after its
  /// driver and fails rather than converging over the refusal.
  bool wasLimitReached() const { return limitReached; }

private:
  struct Frame {
    Operation *parent;
    Attribute templateKey;
  };

  DenseMap<Operation *, Frame> frames;
  unsigned maxDepth = 0;
  bool limitReached = false;
};

// Aggregates memoization for both impl resolution and proof creation.
struct ProofResolutionMemo {
  // Maps a concrete trait application to the canonical proof symbol
  // (either an ImplOp's symbol for self-proofs, or a ProofOp symbol).
  llvm::DenseMap<TraitApplicationAttr, FlatSymbolRefAttr> proofMemo;

  // Tracks impl resolution results to avoid redundant analysis.
  ResolutionMemo resolutionMemo;
};

/// ImplResolver coordinates trait impl resolution and proof construction
/// within a given ModuleOp.
///
/// On construction, it discovers all loaded dialects that provide the
/// `GenerateImplsInterface` and asks them to populate its internal
/// `ImplGeneratorSet`. These generators are used to synthesize or
/// discover implementations when resolving trait claims.
///
/// The main entry point is `resolveAndEnsureProofFor`, which guarantees
/// that a canonical proof exists for a fully-concrete trait application.
/// Resolution proceeds by:
///   1. Returning the symbol of a self-proving `trait.impl` if one exists.
///   2. Otherwise, recursively resolving and ensuring proofs for all
///      requirements and assumptions, then creating or reusing a
///      `trait.proof` operation.
/// Memoization is used to avoid redundant resolution work and to ensure
/// canonicalization of proofs across calls.
///
/// This class may mutate the IR (e.g. by inserting `trait.proof` or `trait.impl` ops)
/// through the provided `OpBuilder`. It only builds ops, never erases or
/// replaces them, so no rewriter capability is required. Callers running under
/// a greedy pattern driver must still hand down that driver's active
/// `PatternRewriter`, whose listener enqueues the inserted ops for the driver
/// to revisit.
class ImplResolver {
  public:
    /// Creates a new `ImplResolver` for the given `module`, recording the
    /// demands it declines to serve in `ledger`.
    /// Finds all loaded dialects that provide the `GenerateImplsInterface` and
    /// populates this `ImplResolver`'s `ImplGeneratorsSet`.
    ///
    /// The ledger is held by shared pointer because this resolver is moved out
    /// of the sub-phase that builds it, and the thread-local sink installed
    /// over both sub-phases points at the ledger's address.
    ImplResolver(ModuleOp module, std::shared_ptr<DemandLedger> ledger);

    /// The demands this resolver's stage declined to serve.
    DemandLedger &getDemandLedger() const { return *ledger; }

    /// Ensures canonical proof for a fully-concrete trait application `claim`.
    /// Resolution proceeds as follows:
    ///   1. If an unconditional ImplOp exists, return its symbol directly.
    ///   2. Otherwise, recursively resolve and ensure proofs for all requirements
    ///      and assumptions, then create (or reuse) a `trait.proof` op and return
    ///      its symbol.
    /// This function may mutate the IR via `builder`.
    ///
    /// Returns the symbol (ImplOp or ProofOp) that proves `claim`, or failure if
    /// no unique and satisfiable impl can be found.
    ///
    /// `refusedOn`, when given, receives the arm impl selection refused this
    /// claim's application on, and is left alone where selection did not
    /// refuse -- a proof that fails downstream of a selected impl names no arm.
    FailureOr<FlatSymbolRefAttr> resolveAndEnsureProofFor(ClaimType claim,
                                                          OpBuilder &builder,
                                                          llvm::function_ref<InFlightDiagnostic()> err = nullptr,
                                                          std::optional<RefutationArm> *refusedOn = nullptr);

    /// Resolves a concrete ProjectionType to the type it projects to.
    /// Uses the internal impl resolution pipeline to find the matching impl,
    /// then looks up the associated type binding and applies substitution.
    ///
    /// `refusedOn`, when given, receives the arm impl selection refused this
    /// projection's application on, and is left alone when selection did not
    /// refuse -- a resolution that fails downstream of a selected impl names no
    /// arm.
    FailureOr<Type> resolveProjectionType(ProjectionType proj,
                                          OpBuilder &builder,
                                          llvm::function_ref<InFlightDiagnostic()> err = nullptr,
                                          std::optional<RefutationArm> *refusedOn = nullptr);

    /// What putting one demand to impl selection settled.
    enum class DemandDisposition : uint8_t {
      /// Selection resolved the projection.
      Served,
      /// Selection refused on the arm no later resolution overturns: two or
      /// more candidates satisfy the application, and candidates are only
      /// appended.
      Refused,
      /// Selection did not serve it, on facts a later resolution may move.
      Deferred,
    };

    /// Puts `demand` to impl selection and says what that settled.
    ///
    /// Whether asking again could ever answer differently is what a caller
    /// scheduling rounds needs and what a bare resolution result does not say.
    /// A claim is served by proving it, which mints the proof its demander
    /// could only read; the two dispositions are read off the same refutation
    /// arm.
    DemandDisposition serveDemand(ProjectionType demand, OpBuilder &builder);
    DemandDisposition serveDemand(ClaimType demand, OpBuilder &builder);

    /// How many facts impl selection has minted: one for each impl it generated
    /// and one for each proof it recorded.
    ///
    /// A refusal stands until the facts it was derived from move, and this is
    /// the monotone quantity that says they have. It counts writes rather than
    /// entries, so an optimistic proof entry a failed recursion takes back out
    /// still counts: a quantity that fell could show a reader the same number
    /// across a fact base that had changed in between.
    uint64_t getFactEpoch() const { return factEpoch; }

    /// How many times what a read of this resolver answers from has changed.
    ///
    /// A read serves from the selections and the proofs recorded so far and
    /// from the module those name, so two reads taken at one value of this
    /// answer alike, and a caller holding an answer knows it still stands while
    /// this stands. It moves wherever the fact base moves, and also where
    /// nothing is minted and a read still gains an answer: selection settling an
    /// application the module already had the impl for, and the commit
    /// respelling what a proof is read through.
    ///
    /// Refusing an application is not such a change, and neither is forgetting
    /// the refusal again. A read fails on a refused application exactly as it
    /// fails on one selection has never been asked about, so writing the entry
    /// and dropping it both leave every answer a read gives where it stood. What
    /// they move is what asking selection itself would have to derive, which is
    /// the negative memo's business and not this quantity's.
    ///
    /// It counts writes rather than entries, for the same reason the fact epoch
    /// does: a count that fell could show a reader the same number across a
    /// record that had changed in between.
    uint64_t getRecordEpoch() const { return recordEpoch; }

    /// Walks `ty` and replaces every concrete (monomorphic) ProjectionType
    /// with its resolved type via full impl lookup.  Polymorphic projections
    /// are left untouched.  Returns the rewritten type.
    Type resolveProjectionsIn(Type ty, OpBuilder &builder);

    /// A replacer that respells every unproven claim whose trait application
    /// this resolver has recorded a proof for.
    ///
    /// The replacer reads the memo rather than copying it, so it answers for
    /// the memo as it stands each time it is asked. A caller must therefore not
    /// record a proof while a replacer is in use: a replacer caches the answers
    /// it has already given, so a memo that grew mid-sweep would respell some
    /// occurrences of a claim and leave others alone. The replacer asserts that
    /// precondition on every answer.
    AttrTypeReplacer makeProvenClaimReplacer() const;

    /// How many trait applications this resolver has recorded a proof for.
    size_t getRecordedProofCount() const { return memo.proofMemo.size(); }

    /// How many trait applications this resolver has recorded an impl selection
    /// for, the record a ground projection resolves through.
    size_t getRecordedImplCount() const {
      return memo.resolutionMemo.chosen.size();
    }

    /// The proof derivations completed over this resolver's span.
    ///
    /// Derivation is a computation over the module's facts and not a fact of
    /// its own, so this is a cache rather than part of the record: a reader
    /// holding this resolver through a handle that may not resolve may still
    /// serve from it and still hold what it derives.
    ProofDerivationMemo &getDerivationMemo() const { return derivations; }

    /// The template instantiations cut over this resolver's span, which is one
    /// stage run. Like the derivation memo this is a computation over the
    /// module rather than a fact of its own, so a reader holding this resolver
    /// through a handle that may not resolve still records into it.
    InstantiationChain &getInstantiationChain() const { return instantiations; }

    /// Says a sweep has respelled the module's copy of the recorded facts,
    /// `replacer` being the rewrite it applied.
    ///
    /// A sweep records no proof, so the fact count does not move for it; what
    /// a derivation reads are spellings, so what was derived before the sweep
    /// was derived from a module that no longer stands. The memo of spelling
    /// pairs answers that by holding nothing across the sweep; the record of
    /// per-application closures is transcribed instead, because it is what a
    /// reader that must not derive serves from and dropping it would leave that
    /// reader with nothing.
    void noteRespelling(AttrTypeReplacer &replacer) const {
      derivations.getClosures().respellWith(replacer);
      derivations.noteRespelling();
      ++recordEpoch;
    }

    /// Forgets every refusal a later resolution could answer differently.
    ///
    /// Selection refuses on two arms and only one of them can move. A refusal
    /// for want of a satisfiable candidate is one an impl generated since can
    /// overturn, and the application it was recorded under is a spelling that
    /// moves too -- respelling a proven claim inside an application's arguments
    /// makes a different application -- so the entry is dropped rather than
    /// re-keyed. A refusal for two or more satisfiable candidates cannot be
    /// overturned: candidates are only ever appended, so a partition that
    /// already had two of them keeps at least two, and re-deriving it would
    /// refuse again at the price of the whole partition.
    void forgetRetriableRefusals();

    /// Whether impl selection is part-way through no application.
    bool isQuiescent() const { return memo.resolutionMemo.visiting.empty(); }

  private:
    friend class ImplGenerationFreeze;
    friend class ReadOnlyImplResolver;

    /// Finds the unique impl for the wanted claim and returns the normalized
    /// claim that was actually used for selection. `refusedOn`, when given,
    /// receives the arm a refusal was refused on.
    FailureOr<ResolvedImpl> resolveImplFor(
        ClaimType wanted,
        OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr,
        std::optional<RefutationArm> *refusedOn = nullptr);

    /// Records `sym` as what proves `app`, counting the fact.
    void recordProof(TraitApplicationAttr app, FlatSymbolRefAttr sym) {
      memo.proofMemo[app] = sym;
      noteFactWritten();
    }

    /// Counts one fact write, so that what was derived from the fact base
    /// before it is no longer an answer about the fact base after it.
    void noteFactWritten() {
      ++factEpoch;
      noteRecordWritten();
      derivations.noteFactWritten();
    }

    /// Counts one write to what a read answers from, whether or not it minted
    /// a fact.
    void noteRecordWritten() { ++recordEpoch; }

    /// Checks whether all of `impl`'s where-clause assumptions are satisfiable
    /// when specialized for `concreteSelf`.
    LogicalResult assumptionsSatisfiableFor(ImplOp impl,
                                            ClaimType concreteSelf,
                                            OpBuilder &builder);

    /// The generators impl selection asks when no candidate impl satisfies a
    /// claim: this resolver's own set, or whatever stands in for them while
    /// something is installed over a span.
    const ImplGenerator &getImplGenerators() const {
      return installedOverride ? *installedOverride : generators;
    }

    mutable ModuleOp module;
    std::shared_ptr<DemandLedger> ledger;
    ProofResolutionMemo memo;
    mutable ProofDerivationMemo derivations;
    mutable InstantiationChain instantiations;
    ImplGeneratorSet generators;
    const ImplGenerator *installedOverride = nullptr;
    uint64_t factEpoch = 0;
    mutable uint64_t recordEpoch = 0;
};

/// Stands in for a resolver's impl generators over a span in which no impl may
/// be generated, and fails the compilation where impl selection asks for one.
///
/// A span forbids generation when the work that would have to see a generated
/// impl has already run: an impl built after that point reaches nothing that
/// was waiting for it, and what the span produces is quietly incomplete rather
/// than loudly wrong. A freeze names the claim that was demanded and the span
/// whose contract the demand broke, at the point selection asked.
///
/// The stage stands one over its instantiation driver, whose patterns read the
/// facts earlier steps recorded and put nothing to selection, so an ask from
/// under it is a component reaching past the record it is meant to read.
class ImplGenerationFreeze : public ImplGenerator {
public:
  /// Installs itself as `resolver`'s generators until it goes out of scope.
  /// `span` names the work whose contract forbids generation, and is what the
  /// failure reports as broken.
  ImplGenerationFreeze(ImplResolver &resolver, StringRef span);
  ~ImplGenerationFreeze();

  ImplGenerationFreeze(const ImplGenerationFreeze &) = delete;
  ImplGenerationFreeze &operator=(const ImplGenerationFreeze &) = delete;

  /// Always fails: being asked to generate at all is the fault this reports,
  /// through a diagnostic at the demanded trait rather than a process abort.
  FailureOr<ImplOp> generateImpl(TraitOp trait,
                                 ClaimType wanted,
                                 OpBuilder &builder) const override;

  /// Whether generation was demanded across this freeze's span. A greedy driver
  /// treats the failure the ask returns as a pattern that did not apply and
  /// keeps converging, so the span's owner reads this after the driver and
  /// fails the stage: the ask emitted its diagnostic, and the stage must not
  /// report success over a span whose contract was broken.
  bool wasAsked() const { return generationAsked; }

private:
  ImplResolver &resolver;
  std::string span;
  const ImplGenerator *displaced;
  mutable bool generationAsked = false;
};

/// A read of one resolver's recorded facts, for a caller that must serve from
/// what impl selection has already settled.
///
/// What this handle withholds is the generator arm: a caller reading through it
/// cannot make impl selection run, so no impl is generated and no proof is
/// minted on its account. The facts themselves are not frozen -- whoever holds
/// the resolver goes on recording selections and creating `trait.proof` ops --
/// so an answer here is what the memo held when it was asked.
///
/// Impl selection keys its memo by the claim whose projections it resolved, so
/// an application asked about here is one spelled as selection recorded it: a
/// caller holding a source spelling with a projection still in it misses.
class ReadOnlyImplResolver {
public:
  explicit ReadOnlyImplResolver(const ImplResolver &resolver)
      : resolver(resolver) {}

  /// What impl selection settled on for `app`: the impl it chose, or the arm it
  /// refused on. Nothing when selection has not been asked about `app`.
  inline std::optional<ResolutionOutcome>
  getRecordedOutcome(TraitApplicationAttr app) const {
    const auto &chosen = resolver.memo.resolutionMemo.chosen;
    auto it = chosen.find(app);
    if (it == chosen.end())
      return std::nullopt;
    return it->second;
  }

  /// The proof derivations completed over the resolver's span. Serving from
  /// them and holding what is derived through them takes no generator arm.
  ProofDerivationMemo &getDerivationMemo() const {
    return resolver.getDerivationMemo();
  }

  /// The template instantiations cut over the resolver's span. Cutting one
  /// takes no generator arm either: the template is already in the module.
  InstantiationChain &getInstantiationChain() const {
    return resolver.getInstantiationChain();
  }

  /// How many times what this reads from has changed. Every answer here is read
  /// off what selection has settled, so two reads taken at one value of this
  /// answer alike.
  uint64_t getRecordEpoch() const { return resolver.getRecordEpoch(); }

  /// The symbol proving `app` -- an impl's own for a self-proof, a
  /// `trait.proof`'s otherwise. Nothing when no proof of `app` is recorded.
  inline std::optional<FlatSymbolRefAttr>
  getRecordedProof(TraitApplicationAttr app) const {
    const auto &proofs = resolver.memo.proofMemo;
    auto it = proofs.find(app);
    if (it == proofs.end())
      return std::nullopt;
    return it->second;
  }

  /// Declines `demand`, recording it as one this read did not serve, and fails.
  ///
  /// A caller that declines leaves the demanded projection or claim spelled as
  /// written, which is what a reader of the recorded demand finds when it asks
  /// whether an unserved demand is still there to serve.
  LogicalResult decline(ProjectionType demand) const;
  LogicalResult decline(ClaimType demand) const;

  /// The impl selection settled on for `wanted`, paired with the claim it
  /// settled it under. Fails where selection was never asked, where it refused,
  /// and where a projection in `wanted` cannot be resolved from what is
  /// recorded.
  ///
  /// Selection keys what it records by the claim whose projections it resolved,
  /// so the source spelling is put through the same resolution before it is
  /// looked up.
  FailureOr<ResolvedImpl> getRecordedImplFor(ClaimType wanted) const;

  /// The type `proj` projects to, from what impl selection has recorded.
  ///
  /// Reading the associated-type binding off the selected impl and specializing
  /// it for the claim selection settled under are reads of the module, so all
  /// that separates this from the resolution it stands in for is where the impl
  /// comes from.
  FailureOr<Type> resolveProjectionType(ProjectionType proj) const;

  /// Walks `ty` and replaces every monomorphic projection this read can
  /// resolve, leaving the rest spelled as written and recording each one.
  Type resolveProjectionsIn(Type ty) const;

  /// The symbol proving `claim`, from what impl selection has recorded. Fails
  /// where no proof of it is recorded, which is what a caller declines on.
  ///
  /// Proofs are recorded under the monomorphic application the selected impl's
  /// self-claim substitution produces, so a source spelling is put through the
  /// same two steps -- selection's own projection resolution, then that
  /// substitution -- before it is looked up.
  FailureOr<FlatSymbolRefAttr> getRecordedProofFor(ClaimType claim) const;

private:
  const ImplResolver &resolver;
};

/// A normalizer over what impl selection has settled, and then over the impls
/// the module holds: `ReadOnlyImplResolver::resolveProjectionsIn` as a callable.
///
/// This is the reading every step that rebuilds an impl's header runs. A trait
/// with two impls whose headers could each bind one application is a trait the
/// module alone answers nothing about -- only the record says which of them
/// selection chose -- so a header spelling a projection over such an
/// application reaches the claim it was chosen for through this and through
/// nothing weaker. A read takes no generator arm, so nothing is minted on its
/// account.
class RecordedProjectionLookup {
public:
  explicit RecordedProjectionLookup(const ImplResolver &resolver)
      : reading(resolver) {}
  explicit RecordedProjectionLookup(const ReadOnlyImplResolver &reading)
      : reading(reading) {}

  FailureOr<Type> operator()(Type ty) const {
    return reading.resolveProjectionsIn(ty);
  }

private:
  ReadOnlyImplResolver reading;
};

} // end mlir::trait
