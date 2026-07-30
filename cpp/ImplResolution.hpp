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
  // When generation runs underneath a greedy pattern driver, the caller must
  // pass that driver's active PatternRewriter itself rather than a builder of
  // its own: the builder's listener is what places generated ops on the
  // driver's worklist, so a builder without that listener silently changes
  // which ops the driver revisits.
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
    FailureOr<Type> resolveProjectionType(ProjectionType proj,
                                          OpBuilder &builder,
                                          llvm::function_ref<InFlightDiagnostic()> err = nullptr);

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
    /// Finds the unique impl for the wanted claim and returns the normalized
    /// claim that was actually used for selection.
    FailureOr<ResolvedImpl> resolveImplFor(
        ClaimType wanted,
        OpBuilder &builder,
        llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// Checks whether all of `impl`'s where-clause assumptions are satisfiable
    /// when specialized for `concreteSelf`.
    LogicalResult assumptionsSatisfiableFor(ImplOp impl,
                                            ClaimType concreteSelf,
                                            OpBuilder &builder);

    mutable ModuleOp module;
    std::shared_ptr<DemandLedger> ledger;
    ProofResolutionMemo memo;
    ImplGeneratorSet generators;
};

} // end mlir::trait
