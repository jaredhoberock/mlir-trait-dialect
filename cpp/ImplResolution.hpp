// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <memory>
#include <variant>

namespace mlir::trait {

// Interface to generate a new ImplOp for the wanted claim, or fail
struct ImplGenerator {
  virtual ~ImplGenerator() = default;

  // Creates exactly one new ImplOp for the wanted claim, or fails.
  // Upon success, returns the newly created ImplOp whose self claim, read for
  // the arguments its own parameters take, must rebuild wanted.
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

/// Why impl selection refused an application, for a caller that reports the
/// refusal somewhere other than where selection was asked.
///
/// The satisfiable candidates ARE the ambiguity, so a refusal on that arm names
/// them; the other arm has none to name. They are what selection had in hand
/// when it refused, so a caller handed an application selection had already
/// refused is given the arm alone.
struct Refutation {
  RefutationArm arm;
  SmallVector<ImplOp> satisfiable;
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

/// A trait application as read in one module.
///
/// A spelling names its symbols in one symbol table, and two modules can spell
/// one application and mean two different impls of it, so what selection
/// settles is settled for that application under the module it was demanded in
/// and not for the spelling alone.
using ScopedApplication = std::pair<Operation *, TraitApplicationAttr>;

// Memoization state for pure impl resolution (no IR mutations).
struct ResolutionMemo {
  // Maps a fully-concrete trait application, as read in one module, to the impl
  // selected for it, or to the arm on which selection was refused when no
  // unique impl exists.
  DenseMap<ScopedApplication, ResolutionOutcome> chosen;

  // The applications impl selection is part-way through, outermost first. A
  // repeat is a resolution cycle; the number of frames is how deep the
  // obligation chain has recursed, which is what bounds a chain whose every
  // step is a new application.
  SmallVector<ObligationFrame> visiting;

  // A memo for assumptionsSatisfiableFor
  // For every (ImplOp, TraitApplicationAttr) in this set, the ImplOp's
  // assumptions are known to be satisfiable for the given TraitApplicationAttr
  // We only memoize satisfiable results because new proofs appear in the IR
  // as resolution unfolds
  DenseSet<std::pair<ImplOp,TraitApplicationAttr>> assumptionsKnownSatisfiable;

  // The applications a generator has already supplied an impl for, in the
  // module it was supplied into. Generation supplies an impl the module lacks,
  // and a generated impl is a function of the application it was asked for, so
  // asking twice would publish a second op under the name the first already
  // holds. The impl supplied stands in the module, where the candidate scan
  // reads it, so a later round judges it against the facts as they then stand
  // without generation running again. Unlike a retriable refusal, this outlives
  // the flush: what the module holds is not a question anything reopens.
  DenseSet<ScopedApplication> generatedFor;
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

/// One step of the resolution of a projection with a ground head: the impl
/// selection settled on for the projection's application, the arguments that
/// impl's parameters take at the claim selection chose it for, and the impl's
/// binding of the projected associated type at those arguments and at the
/// projection's own associated-type arguments as spelled.
///
/// The binding is specialized and nothing more: a projection it spells, the
/// impl's own or one an associated-type argument carried in, is a step of its
/// own. So `projection = binding` is exactly what the witness verifier's
/// specialization of the cited impl's binding reproduces. The only constructor
/// asks selection about the projection's own application, reads the arguments
/// off what selection settled, and specializes the binding itself, so a step
/// that exists states what the impl selected for its projection binds at the
/// arguments it takes there. Only the two owners of selection and its record
/// construct one.
class ProjectionResolution {
public:
  ProjectionType getProjection() const { return projection; }
  ImplOp getImpl() const { return impl; }
  const SpecializationMap &getArguments() const { return arguments; }
  Type getBinding() const { return binding; }

private:
  friend class ImplResolver;
  friend class ReadOnlyImplResolver;

  /// The step resolving `projection` through the impl `select` settles on for
  /// the projection's application, read through `record`, the context
  /// selection chose it under. Fails where selection settles on no impl, where
  /// the impl's header does not carry to the claim selection chose it for, or
  /// where the impl binds no such associated type at the projection's
  /// associated-type arguments.
  static FailureOr<ProjectionResolution>
  get(ProjectionType projection,
      llvm::function_ref<FailureOr<ResolvedImpl>(ClaimType)> select,
      const ReadOnlyImplResolver &record,
      llvm::function_ref<InFlightDiagnostic()> err = nullptr);

  ProjectionResolution(ProjectionType projection, ImplOp impl,
                       SpecializationMap arguments, Type binding)
      : projection(projection), impl(impl), arguments(std::move(arguments)),
        binding(binding) {}

  ProjectionType projection;
  ImplOp impl;
  SpecializationMap arguments;
  Type binding;
};

/// Where the evidence for a monomorphic equality reads the facts it cites:
/// `hop` resolves one step of a monomorphic projection and `proofOf` proves an
/// application claim, answering it proven. Each fails where it does not serve.
/// `module` bounds the fixed-point resolution of an endpoint.
struct EqualitySource {
  llvm::function_ref<FailureOr<ProjectionResolution>(ProjectionType)> hop;
  llvm::function_ref<FailureOr<ClaimType>(ClaimType)> proofOf;
  ModuleOp module;
};

struct EqualityResolution;

/// One step of an endpoint's resolution, as its witness cites it: the equality
/// `projection = binding`, the impl selection chose for the projection's
/// application, and the evidence for each of that impl's where entries at the
/// arguments its parameters take there, in order -- a proven application at an
/// application entry, and the resolution of the equality at an equality entry.
struct ResolutionStep {
  TypeEqualityAttr equality;
  FlatSymbolRefAttr impl;
  SmallVector<std::variant<ClaimType, std::shared_ptr<EqualityResolution>>>
      premises;
};

/// The resolution of a monomorphic equality's two sides to one spelling: one
/// step per distinct projection resolved on the way.
struct EqualityResolution {
  TypeEqualityAttr equality;
  SmallVector<ResolutionStep> steps;
};

/// The ground spellings the sides of `eq` resolve to through `source`,
/// appending one step per distinct ground projection resolved on the way to
/// `steps`; identical sides are read as spelled. The walk descends composites,
/// so a projection nested inside one yields its step just as a top-level
/// projection does, and runs to a fixed point because a binding may itself
/// spell a projection. An equality entry of a resolving impl is resolved the
/// same way, and must reach one spelling. Fails where a step fails or a ground
/// projection still stands.
FailureOr<std::pair<Type, Type>>
resolveEquality(TypeEqualityAttr eq, const EqualitySource &source,
                SmallVectorImpl<ResolutionStep> &steps, unsigned depth = 0);

/// Builds at `builder`'s insertion point the evidence for `eq` from `steps`,
/// the steps resolving its sides to one ground spelling: refl for identical
/// sides; the sole step's witness where it proves `eq` as spelled; else the
/// composition of every step's witness, whose ground congruence closure carries
/// the sides together across every step. Each step's witness cites its impl
/// with one claim per where entry, a witness of the proof at an application
/// entry and the evidence built for the equality at an equality entry. The
/// result is closed: it reads no value from around it.
Value buildEqualityEvidence(OpBuilder &builder, Location loc,
                            TypeEqualityAttr eq,
                            ArrayRef<ResolutionStep> steps);

/// The template instantiations one stage run has cut.
///
/// Each instance is cut for a template at a call standing inside another
/// function, so the instances form a forest and what matters about one is how
/// many instances of the SAME template stand on the path that reaches it. A
/// template that instantiates itself at a larger type mints a distinct instance
/// at every step, so no cycle guard sees a repeat; that count is what tells
/// such a chain from a deep but finite nest of distinct templates, which Rust's
/// monomorphization collector counts the same way, per function.
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
  /// `parent`.
  ///
  /// An instance already recorded keeps the chain it was first cut on, and an
  /// instance that is its own parent -- a call whose specialization reached the
  /// very function it stands in -- records nothing, so the chain stays a
  /// forest.
  void note(Operation *instance, Operation *parent, Attribute templateKey);

  /// The frames from the root down to `instance`, each a function and the
  /// template it was cut for. This is what a refusal at the depth limit names.
  SmallVector<std::pair<Operation *, Attribute>>
  chainTo(Operation *instance) const;

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
  bool limitReached = false;
};

// Aggregates memoization for both impl resolution and proof creation.
struct ProofResolutionMemo {
  // Maps a concrete trait application, as read in one module, to the canonical
  // proof symbol there (either an ImplOp's symbol for self-proofs, or a ProofOp
  // symbol). The symbol is one that module's symbol table resolves, which is
  // why the module is part of the key.
  llvm::DenseMap<ScopedApplication, FlatSymbolRefAttr> proofMemo;

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
///   1. Proving it by a self-proving `trait.impl` if one exists.
///   2. Otherwise, recursively resolving and ensuring proofs for the impl's
///      where entries, then creating or reusing a `trait.proof` operation.
/// Memoization is used to avoid redundant resolution work and to ensure
/// canonicalization of proofs across calls.
///
/// This class may mutate the IR (e.g. by inserting `trait.proof` or `trait.impl` ops)
/// through the provided `OpBuilder`. It only builds ops, never erases or
/// replaces them, so no rewriter capability is required. Callers running under
/// a greedy pattern driver must still hand down that driver's active
/// `PatternRewriter`, whose listener enqueues the inserted ops for the driver
/// to revisit.
///
/// Every ask names the module the demand was read in. That module is the symbol
/// table the demand's spelling names its trait and impls in, the one whose
/// proofs an answer may cite, and the one a generated impl or a created proof
/// belongs in; it is what the records are keyed under, so a demand raised
/// inside a nested module is never answered with a symbol only the module
/// around it resolves. A pass root asking about its own ops names itself.
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
    ///   1. If an unconditional ImplOp exists, its symbol proves the claim.
    ///   2. Otherwise, recursively resolve and ensure proofs for the impl's
    ///      where entries, then create (or reuse) a `trait.proof` op whose body
    ///      derives the claim from the impl over that evidence and whose symbol
    ///      proves it. The trait's requirements are the impl's to return.
    /// This function may mutate the IR via `builder`.
    ///
    /// Returns `claim` proven: the application its proof is recorded under,
    /// which is `claim`'s with its projections resolved as selection resolved
    /// them, naming the symbol (ImplOp or ProofOp) that proves it. Fails if no
    /// unique and satisfiable impl can be found.
    ///
    /// `refusedOn`, when given, receives what impl selection refused this
    /// claim's application on, and is left alone where selection did not
    /// refuse -- a proof that fails downstream of a selected impl names no
    /// refutation.
    FailureOr<ClaimType> resolveAndEnsureProofFor(ClaimType claim,
                                                  ModuleOp scope,
                                                  OpBuilder &builder,
                                                  llvm::function_ref<InFlightDiagnostic()> err = nullptr,
                                                  std::optional<Refutation> *refusedOn = nullptr);

    /// Resolves one step of a concrete ProjectionType: the impl the internal
    /// impl resolution pipeline selects for its application, the arguments
    /// that impl takes there, and its associated-type binding specialized at
    /// them.
    ///
    /// `refusedOn`, when given, receives what impl selection refused this
    /// projection's application on, and is left alone when selection did not
    /// refuse -- a resolution that fails downstream of a selected impl names no
    /// refutation.
    FailureOr<ProjectionResolution> resolveProjection(
        ProjectionType proj, ModuleOp scope, OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr,
        std::optional<Refutation> *refusedOn = nullptr);

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
    /// arm. A projection deferred is recorded in the ledger, and the stage's
    /// exit check (`DemandLedger::checkStandingDemandsServed`) refuses a
    /// recorded demand still spelled and never served when the rounds end.
    DemandDisposition serveDemand(ProjectionType demand, ModuleOp scope,
                                  OpBuilder &builder);
    DemandDisposition serveDemand(ClaimType demand, ModuleOp scope,
                                  OpBuilder &builder);

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

    /// A replacer that respells every unproven claim whose trait application
    /// this resolver has recorded a proof for in `scope`.
    ///
    /// The proof a claim names is a symbol `scope` resolves, so a replacer
    /// serves the ops of one module and a sweep over a module holding others
    /// takes one replacer per module it visits.
    ///
    /// The replacer reads the memo rather than copying it, so it answers for
    /// the memo as it stands each time it is asked. A caller must therefore not
    /// record a proof while a replacer is in use: a replacer caches the answers
    /// it has already given, so a memo that grew mid-sweep would respell some
    /// occurrences of a claim and leave others alone. The replacer asserts that
    /// precondition on every answer.
    AttrTypeReplacer makeProvenClaimReplacer(ModuleOp scope) const;

    /// How many trait applications this resolver has recorded a proof for.
    size_t getRecordedProofCount() const { return memo.proofMemo.size(); }

    /// How many trait applications this resolver has recorded an impl selection
    /// for, the record a ground projection resolves through.
    size_t getRecordedImplCount() const {
      return memo.resolutionMemo.chosen.size();
    }

    /// The template instantiations cut over this resolver's span, which is one
    /// stage run. This is a computation over the module rather than a fact of
    /// its own, so a reader holding this resolver through a handle that may not
    /// resolve still records into it.
    InstantiationChain &getInstantiationChain() const { return instantiations; }

    /// Says a sweep has respelled the module's copy of the recorded facts.
    ///
    /// A sweep records no proof, so the fact count does not move for it; what
    /// a read answers from are spellings, so the record epoch moves.
    void noteRespelling() const { ++recordEpoch; }

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

    /// The proof standing in `scope` whose body derives `app` from `impl`
    /// given, at its application entries in order, the proofs `subproofs`
    /// names; null where none stands. An equality entry is ground and has one
    /// answer, so it identifies nothing.
    ClaimType findProof(ModuleOp scope, ImplOp impl, TraitApplicationAttr app,
                        ArrayRef<FlatSymbolRefAttr> subproofs) const;

    /// Writes at the end of `scope` the proof whose body derives `app` from
    /// `impl` at `arguments` over one premise per entry of `entries`, `impl`'s
    /// where entries at those arguments: a witness of the proof `subproofs`
    /// names for each application entry, in order, and for each equality entry
    /// the evidence its `equalitySteps` build. This is a derive transcribed,
    /// not a selection: it records nothing a selection reads. The proof is
    /// named `name` where given, a name reserved for it (`freeProofName`), and
    /// otherwise by its impl and arguments, told apart by its subproofs where
    /// that name is taken.
    ClaimType writeProof(ModuleOp scope, ImplOp impl, TraitApplicationAttr app,
                         const SpecializationMap &arguments,
                         ArrayRef<ClaimType> entries,
                         ArrayRef<FlatSymbolRefAttr> subproofs,
                         ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
                         OpBuilder &builder, StringAttr name = {}) const;

  private:
    friend class ImplGenerationFreeze;
    friend class ReadOnlyImplResolver;

    /// Walks `ty` and replaces every concrete (monomorphic) ProjectionType
    /// with its resolved type via full impl lookup.  Polymorphic projections
    /// are left untouched.  Returns the rewritten type.
    Type resolveProjectionsIn(Type ty, ModuleOp scope, OpBuilder &builder);

    /// Finds the unique impl for the wanted claim and returns the normalized
    /// claim that was actually used for selection. `refusedOn`, when given,
    /// receives the refutation a refusal was refused on.
    FailureOr<ResolvedImpl> resolveImplFor(
        ClaimType wanted,
        ModuleOp scope,
        OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr,
        std::optional<Refutation> *refusedOn = nullptr);

    /// Records `sym` as what proves `app` in `scope`, counting the fact, and
    /// answers the claim of `app` that `sym` proves.
    ClaimType recordProof(ModuleOp scope, TraitApplicationAttr app,
                          FlatSymbolRefAttr sym) {
      memo.proofMemo[{scope, app}] = sym;
      noteFactWritten();
      return ClaimType::get(scope.getContext(), app, sym);
    }

    /// Counts one fact write, so that what was derived from the fact base
    /// before it is no longer an answer about the fact base after it.
    void noteFactWritten() {
      ++factEpoch;
      noteRecordWritten();
    }

    /// Counts one write to what a read answers from, whether or not it minted
    /// a fact.
    void noteRecordWritten() { ++recordEpoch; }

    /// The proofs standing in one module: by the impl each stands over and the
    /// application it proves, and by name. Read off the module at the first
    /// mint in it and extended by every proof minted there, which is the one
    /// site that writes a proof while the stage runs. Nothing the stage runs
    /// erases a proof -- a rewrite driver never takes a symbol for dead, and
    /// proofs go only in the erase pass after the stage -- so every op held
    /// here stands.
    struct StandingProofs {
      /// Every proof of one impl at one application, in module order; two
      /// stand apart where their derives are given different premises.
      DenseMap<std::pair<ImplOp, TraitApplicationAttr>, SmallVector<ProofOp, 1>>
          byClaim;
      DenseMap<StringAttr, ProofOp> byName;

      /// Adds `proof`.
      void note(ProofOp proof);
    };

    /// The proofs standing in `scope`, read once. A view of the module the
    /// stage extends wherever it writes a proof, so a reader holding the
    /// resolver read-only still keeps it current.
    StandingProofs &getStandingProofs(ModuleOp scope) const;

    /// Checks whether all of `impl`'s where-clause assumptions are satisfiable
    /// when specialized for `concreteSelf`, read in `scope`.
    LogicalResult assumptionsSatisfiableFor(ImplOp impl,
                                            ClaimType concreteSelf,
                                            ModuleOp scope,
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
    mutable DenseMap<Operation *, StandingProofs> standingProofs;

    /// `base` where no symbol of `scope` holds it and no proof being proven
    /// there has reserved it; otherwise `base` told apart by a hash of `salt`,
    /// rehashed until free. Mangled names are not one-to-one -- an impl may be
    /// named what another's mangling at some arguments spells, and any symbol
    /// may be -- so a name is free only once both are asked.
    StringAttr freeProofName(ModuleOp scope, StringRef base,
                             StringRef salt) const;

    /// The names reserved in each scope for proofs whose premises are being
    /// proven, which a premise may cite before the proof is written.
    mutable DenseSet<std::pair<Operation *, StringAttr>> reservedProofNames;
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
/// selected on its account; the one proof it writes is the one a derive states
/// (`writeProof`). The facts themselves are not frozen -- whoever holds
/// the resolver goes on recording selections and creating `trait.proof` ops --
/// so an answer here is what the memo held when it was asked.
///
/// Impl selection keys its memo by the claim whose projections it resolved, so
/// an application asked about here is one spelled as selection recorded it: a
/// caller holding a source spelling with a projection still in it misses.
///
/// A read is a read in one module: what selection settled is settled for an
/// application under the module it was demanded in, and the symbol an answer
/// names is one that module's symbol table resolves. A reader built from a
/// resolver alone reads in the module that resolver was built for; a caller
/// holding an op reads in the op's own module, which `in` hands it.
class ReadOnlyImplResolver {
public:
  explicit ReadOnlyImplResolver(const ImplResolver &resolver)
      : resolver(resolver), scope(resolver.module) {}

  ReadOnlyImplResolver(const ImplResolver &resolver, ModuleOp scope)
      : resolver(resolver), scope(scope) {}

  /// This read taken in `scope` instead.
  ReadOnlyImplResolver in(ModuleOp scope) const {
    return ReadOnlyImplResolver(resolver, scope);
  }

  /// What impl selection settled on for `app` here: the impl it chose, or the
  /// arm it refused on. Nothing when selection has not been asked about `app`
  /// in this module.
  inline std::optional<ResolutionOutcome>
  getRecordedOutcome(TraitApplicationAttr app) const {
    const auto &chosen = resolver.memo.resolutionMemo.chosen;
    auto it = chosen.find({scope, app});
    if (it == chosen.end())
      return std::nullopt;
    return it->second;
  }

  /// The template instantiations cut over the resolver's span. Cutting one
  /// takes no generator arm: the template is already in the module.
  InstantiationChain &getInstantiationChain() const {
    return resolver.getInstantiationChain();
  }

  /// The proof standing here that a derive states
  /// (`ImplResolver::findProof`).
  ClaimType findProof(ImplOp impl, TraitApplicationAttr app,
                      ArrayRef<FlatSymbolRefAttr> subproofs) const {
    return resolver.findProof(scope, impl, app, subproofs);
  }

  /// The proof a derive states, written here (`ImplResolver::writeProof`).
  ClaimType writeProof(ImplOp impl, TraitApplicationAttr app,
                       const SpecializationMap &arguments,
                       ArrayRef<ClaimType> entries,
                       ArrayRef<FlatSymbolRefAttr> subproofs,
                       ArrayRef<SmallVector<ResolutionStep>> equalitySteps,
                       OpBuilder &builder) const {
    return resolver.writeProof(scope, impl, app, arguments, entries, subproofs,
                               equalitySteps, builder);
  }

  /// How many times what this reads from has changed. Every answer here is read
  /// off what selection has settled, so two reads taken at one value of this
  /// answer alike.
  uint64_t getRecordEpoch() const { return resolver.getRecordEpoch(); }

  /// The symbol proving `app` here -- an impl's own for a self-proof, a
  /// `trait.proof`'s otherwise. Nothing when no proof of `app` is recorded in
  /// this module.
  inline std::optional<FlatSymbolRefAttr>
  getRecordedProof(TraitApplicationAttr app) const {
    const auto &proofs = resolver.memo.proofMemo;
    auto it = proofs.find({scope, app});
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

  /// One step of `proj`'s resolution, from what impl selection has recorded.
  ///
  /// Reading the associated-type binding off the selected impl and specializing
  /// it for the claim selection settled under are reads of the module, so all
  /// that separates this from the resolution it stands in for is where the impl
  /// comes from.
  FailureOr<ProjectionResolution> resolveProjection(ProjectionType proj) const;

  /// Walks `ty` and replaces every monomorphic projection this read can
  /// resolve, leaving the rest spelled as written and recording each one.
  Type resolveProjectionsIn(Type ty) const;

  /// `claim` proven, from what impl selection has recorded: the application
  /// its proof is recorded under, naming the symbol that proves it. Fails where
  /// no proof of it is recorded, which is what a caller declines on.
  ///
  /// Proofs are recorded under the monomorphic application the selected impl's
  /// self-claim substitution produces, so a source spelling is put through the
  /// same two steps -- selection's own projection resolution, then that
  /// substitution -- before it is looked up.
  FailureOr<ClaimType> getRecordedProofFor(ClaimType claim) const;

private:
  /// The impl selection settled on for `wanted`, paired with the claim it
  /// settled it under. Fails where selection was never asked, where it refused,
  /// and where a projection in `wanted` cannot be resolved from what is
  /// recorded.
  ///
  /// Selection keys what it records by the claim whose projections it resolved,
  /// so the source spelling is put through the same resolution before it is
  /// looked up.
  FailureOr<ResolvedImpl> getRecordedImplFor(ClaimType wanted) const;

  const ImplResolver &resolver;
  ModuleOp scope;
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
  RecordedProjectionLookup(const ImplResolver &resolver, ModuleOp scope)
      : reading(resolver, scope) {}
  explicit RecordedProjectionLookup(const ReadOnlyImplResolver &reading)
      : reading(reading) {}

  FailureOr<Type> operator()(Type ty) const {
    return reading.resolveProjectionsIn(ty);
  }

private:
  ReadOnlyImplResolver reading;
};

} // end mlir::trait
