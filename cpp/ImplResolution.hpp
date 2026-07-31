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

/// The number of refutation arms, for reporting the partition.
inline constexpr unsigned numRefutationArms = 2;

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

  // Tracks applications currently being resolved to detect resolution cycles.
  DenseSet<TraitApplicationAttr> visiting;

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

    /// Writes the facts this resolver has recorded to the stage-record channel:
    /// a digest over the canonical rendering of all of them, and the counts
    /// behind it.
    ///
    /// The facts live in pointer-keyed maps, whose iteration order is the
    /// allocator's, so the rendering is sorted before it is digested and one
    /// run's digest is comparable with another's. Refusals render as one token
    /// whichever arm they carry: which arm a refusal is refused on decides
    /// whether a later round may retry it, and a digest that moved with that
    /// could not tell a change of retry policy apart from a change of fact.
    void reportRecordedFacts() const;

    /// A digest over the canonical rendering of those same facts, for a reader
    /// comparing what one span of resolution settled against another.
    uint64_t getRecordedFactsDigest() const;

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
    FailureOr<FlatSymbolRefAttr> resolveAndEnsureProofFor(ClaimType claim,
                                                          OpBuilder &builder,
                                                          llvm::function_ref<InFlightDiagnostic()> err = nullptr);

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
    DemandDisposition serveDemand(ProjectionType demand, OpBuilder &builder);

    /// How many facts impl selection has minted: one for each impl it generated
    /// and one for each proof it recorded.
    ///
    /// A refusal stands until the facts it was derived from move, and this is
    /// the monotone quantity that says they have. It counts writes rather than
    /// entries, so an optimistic proof entry a failed recursion takes back out
    /// still counts: a quantity that fell could show a reader the same number
    /// across a fact base that had changed in between.
    uint64_t getFactEpoch() const { return factEpoch; }

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

    /// How many refusals of each kind `forgetRetriableRefusals` found, and what
    /// became of the refusals the call before it forgot.
    struct RefusalCounts {
      /// Refusals dropped, on the arm a later resolution can overturn.
      uint64_t forgotten = 0;
      /// Refusals kept, on the arm no later resolution can overturn.
      uint64_t kept = 0;
      /// Of the refusals the previous call dropped: how many selection has
      /// since answered with an impl, and how many it has refused again.
      uint64_t overturned = 0;
      uint64_t reEarned = 0;
    };

    /// Forgets every refusal a later resolution could answer differently, and
    /// says what became of the ones the call before this one forgot.
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
    ///
    /// Dropping a negative only pays where a later resolution answers
    /// differently, so the counts say how much of the last drop was re-earned
    /// and how much was overturned.
    RefusalCounts forgetRetriableRefusals();

    /// Whether impl selection is part-way through no application.
    bool isQuiescent() const { return memo.resolutionMemo.visiting.empty(); }

    /// Whether every proof this resolver has recorded names an op of the module
    /// that can prove an application: a `trait.proof`, or an unconditional
    /// `trait.impl`, which proves its own self claim.
    ///
    /// Proof creation enters its symbol in the memo before recursing into the
    /// obligations, so that an obligation coming back round to the claim being
    /// proven meets the memo instead of diverging, and takes the entry back out
    /// if that recursion fails. Where no creation is part-way through, every
    /// entry therefore names a proof that exists.
    ///
    /// Answering costs a symbol table over the whole module, which every round
    /// invalidates by rewriting it, so callers ask where the cross-checks are
    /// armed rather than on every compile.
    bool recordsOnlyRealizedProofs() const;

    /// The proof memo as a substitution: every trait application it has a proof
    /// for, mapped from its unproven claim spelling to its proven one.
    ///
    /// Applying this substitution to a type is the same rewrite as asking the
    /// memo about the claims that type spells, at the cost of a copy of the
    /// whole memo whatever is being respelled.
    inline EvidenceBindings buildClaimSubstitutionFromMemo() const {
      MLIRContext* ctx = module.getContext();
      EvidenceBindings subst;
      for (auto [app, proof] : memo.proofMemo) {
        ClaimType unproven = ClaimType::get(ctx, app, nullptr);
        ClaimType proven = ClaimType::get(ctx, app, proof);
        subst.bind(unproven, proven);
      }
      return subst;
    }

  private:
    friend class ImplGenerationFreeze;
    friend class ImplGenerationTally;
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

    /// Counts one fact write, and tells the ledger that what it filed as
    /// derived from the fact base was derived from an earlier one.
    void noteFactWritten() {
      ++factEpoch;
      forgetProofDerivations();
    }

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
    ImplGeneratorSet generators;
    const ImplGenerator *installedOverride = nullptr;
    uint64_t factEpoch = 0;
    /// The applications the last `forgetRetriableRefusals` dropped, so that the
    /// next one can say what became of them.
    SmallVector<TraitApplicationAttr> lastForgotten;
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
/// XXX TODO: a freeze stands over the instantiation driver only where the
/// environment asks for one (see isInstantiationFreezeRequested); a tally
/// stands there otherwise, counting the impls the drivers still ask for. The
/// freeze becomes unconditional once that count is zero on every row. Delete
/// this if the drivers never stop asking.
class ImplGenerationFreeze : public ImplGenerator {
public:
  /// Installs itself as `resolver`'s generators until it goes out of scope.
  /// `span` names the work whose contract forbids generation, and is what the
  /// failure reports as broken.
  ImplGenerationFreeze(ImplResolver &resolver, StringRef span);
  ~ImplGenerationFreeze();

  ImplGenerationFreeze(const ImplGenerationFreeze &) = delete;
  ImplGenerationFreeze &operator=(const ImplGenerationFreeze &) = delete;

  /// Never returns: being asked to generate at all is the fault this reports.
  FailureOr<ImplOp> generateImpl(TraitOp trait,
                                 ClaimType wanted,
                                 OpBuilder &builder) const override;

private:
  ImplResolver &resolver;
  std::string span;
  const ImplGenerator *displaced;
};

/// Setting this in the environment stands a freeze over the stage's
/// instantiation driver, so a module whose driver still generates impls stops
/// there and names the claim it asked for.
///
/// XXX TODO: this is how the freeze is reached while nothing arms it for real.
/// Delete it, with the switch below, once the freeze stands over that span
/// unconditionally and the tally beside it goes.
inline constexpr const char *freezeInstantiationEnvironmentVariable =
    "TRAIT_FREEZE_INSTANTIATION";

/// Whether the environment asked for a freeze over the instantiation driver.
bool isInstantiationFreezeRequested();

/// Counts the impls a span asks impl selection to generate, leaving what the
/// generators do unchanged.
///
/// Generation belongs to whoever schedules resolution, and a span that
/// generates while a pattern driver is running is generating where its
/// scheduler did not plan to. The count says how much of that is left.
///
/// XXX TODO: this measures the generation a pattern driver still does. Delete
/// it, and the fallback it measures, once a freeze stands over the driver span
/// in its place.
class ImplGenerationTally : public ImplGenerator {
public:
  /// Installs itself as `resolver`'s generators until it goes out of scope,
  /// forwarding to whatever it displaced.
  explicit ImplGenerationTally(ImplResolver &resolver);
  ~ImplGenerationTally();

  ImplGenerationTally(const ImplGenerationTally &) = delete;
  ImplGenerationTally &operator=(const ImplGenerationTally &) = delete;

  FailureOr<ImplOp> generateImpl(TraitOp trait,
                                 ClaimType wanted,
                                 OpBuilder &builder) const override;

  /// How many times generation was asked for while this tally stood.
  uint64_t getAsks() const { return asks; }

private:
  ImplResolver &resolver;
  const ImplGenerator &forwardTo;
  const ImplGenerator *displaced;
  mutable uint64_t asks = 0;
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
///
/// XXX TODO: nothing reads through this handle yet. It becomes how the stage's
/// rewrite patterns reach resolution facts once a driver no longer generates
/// impls of its own, which is what the tally over the driver span measures.
/// Delete it if the drivers never stop generating.
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
  /// A caller that declines leaves the demanded projection spelled as written,
  /// which is what a reader of the recorded demand finds when it asks whether
  /// an unserved demand is still there to serve.
  LogicalResult decline(ProjectionType demand) const;

private:
  const ImplResolver &resolver;
};

} // end mlir::trait
