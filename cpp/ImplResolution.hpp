#pragma once

#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include <memory>

namespace mlir::trait {

// Interface to generate a new ImplOp for the wanted claim
// success() -> at least one ImplOp was inserted (IR edited)
// failure() -> not applicable / no edit
struct ImplGenerator {
  virtual ~ImplGenerator() = default;

  virtual LogicalResult
  generateImpl(TraitOp trait,
               ClaimType wanted,
               PatternRewriter &rewriter) const = 0;
};

// Composite that itself behaves like an ImplGenerator
class ImplGeneratorSet : public ImplGenerator {
  public:
    inline LogicalResult
    generateImpl(TraitOp trait,
                 ClaimType wanted,
                 PatternRewriter &rewriter) const override {
      for (const auto &g : generators) {
        if (succeeded(g->generateImpl(trait, wanted, rewriter)))
          return success();
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

// Memoization state for pure impl resolution (no IR mutations).
struct ResolutionMemo {
  // Maps a fully-concrete trait application to its resolved ImplOp
  // (or to failure if resolution was attempted and no unique impl exists).
  DenseMap<TraitApplicationAttr, FailureOr<ImplOp>> chosen;

  // Tracks applications currently being resolved to detect resolution cycles.
  DenseSet<TraitApplicationAttr> visiting;

  // A memo for assumptionsSatisfiableFor
  // For every (ImplOp, TraitApplicationAttr) in this set, the ImplOp's
  // assumptions are known to be satisfiable for the given TraitApplicationAttr
  // We only memoize satisfiable results because new proofs appear in the IR
  // as resolution unfolds
  DenseSet<std::pair<ImplOp,TraitApplicationAttr>> assumptionsKnownSatisfiable;
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
/// through the provided `PatternRewriter`.
class ImplResolver {
  public:
    /// Creates a new `ImplResolver` for the given `module`.
    /// Finds all loaded dialects that provide the `GenerateImplsInterface` and
    /// populates this `ImplResolver`'s `ImplGeneratorsSet`.
    ImplResolver(ModuleOp module);

    /// Ensures canonical proof for a fully-concrete trait application `claim`.
    /// Resolution proceeds as follows:
    ///   1. If a self-proving ImplOp exists, return its symbol directly.
    ///   2. Otherwise, recursively resolve and ensure proofs for all requirements
    ///      and assumptions, then create (or reuse) a `trait.proof` op and return
    ///      its symbol.
    /// This function may mutate the IR via `rewriter`.
    ///
    /// Returns the symbol (ImplOp or ProofOp) that proves `claim`, or failure if
    /// no unique and satisfiable impl can be found.
    FailureOr<FlatSymbolRefAttr> resolveAndEnsureProofFor(ClaimType claim,
                                                          PatternRewriter &rewriter,
                                                          llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    /// Builds a substitution mapping concrete, unproven ClaimTypes to
    /// proven ClaimTypes given the current state of the proof memo
    inline DenseMap<Type,Type> buildClaimSubstitutionFromMemo() const {
      MLIRContext* ctx = module.getContext();
      DenseMap<Type,Type> subst;
      for (auto [app, proof] : memo.proofMemo) {
        ClaimType unproven = ClaimType::get(ctx, app, nullptr);
        ClaimType proven = ClaimType::get(ctx, app, proof);
        subst[unproven] = proven;
      }
      return subst;
    }

  private:
    mutable ModuleOp module;
    ProofResolutionMemo memo;
    ImplGeneratorSet generators;
};

} // end mlir::trait
