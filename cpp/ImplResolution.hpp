#pragma once

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
  generate(TraitOp trait,
           ClaimType wanted,
           PatternRewriter &rewriter) const = 0;
};

// Composite that itself behaves like an ImplGenerator
class ImplGeneratorSet : public ImplGenerator {
  public:
    inline LogicalResult
    generate(TraitOp trait,
             ClaimType wanted,
             PatternRewriter &rewriter) const override {
      for (const auto &g : generators) {
        if (succeeded(g->generate(trait, wanted, rewriter)))
          return success();
      }
      return failure();
    }

    inline ImplGeneratorSet &add(std::unique_ptr<ImplGenerator> g) {
      generators.emplace_back(std::move(g));
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

  // Applications already determined to have at least one satisfiable impl.
  DenseSet<TraitApplicationAttr> knownSatisfiable;
};

// Aggregates memoization for both impl resolution and proof creation.
struct ProofResolutionMemo {
  // Maps a concrete trait application to the canonical proof symbol
  // (either an ImplOp's symbol for self-proofs, or a ProofOp symbol).
  llvm::DenseMap<TraitApplicationAttr, FlatSymbolRefAttr> proofMemo;

  // Tracks impl resolution results to avoid redundant analysis.
  ResolutionMemo resolutionMemo;
};

/// Ensures canonical proof for a fully-concrete trait application `claim`.
/// Resolution proceeds as follows:
///   1. If a self-proving ImplOp exists, return its symbol directly.
///   2. Otherwise, recursively resolve and ensure proofs for all requirements
///      and assumptions, then create (or reuse) a `trait.proof` op and return
///      its symbol.
/// Uses `memo` to:
///   - Avoid redundant impl resolution.
///   - Avoid regenerating existing proofs.
/// This function may mutate the IR via `rewriter`.
///
/// Returns the symbol (ImplOp or ProofOp) that proves `claim`, or failure if
/// no unique and satisfiable impl can be found.
FailureOr<FlatSymbolRefAttr>
resolveAndEnsureProofFor(ClaimType claim,
                         ModuleOp module,
                         ProofResolutionMemo &memo,
                         const ImplGenerator &gen,
                         PatternRewriter &rewriter);

} // end mlir::trait
