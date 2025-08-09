#pragma once

#include "Attributes.hpp"
#include "Ops.hpp"

namespace mlir::trait {

/// Ensures there is canonical evidence for the concrete trait application `app`.
/// - If a self-proving ImplOp exists, returns its symbol.
/// - Otherwise (impl has requirements/assumptions), creates or reuses a
///   `trait.proof` op, recursively ensuring subproofs, and returns the proof's
///   symbol.
/// - Uses `memo` (keyed by TraitApplicationAttr) to avoid rebuilding evidence.
/// - May mutate the IR via `rewriter`.
///
/// Returns the symbol (ImplOp or ProofOp) that proves `app`, or failure on
/// missing/ambiguous implementations.
FailureOr<FlatSymbolRefAttr>
resolveAndEnsureProofFor(TraitApplicationAttr app,
                         ModuleOp module,
                         llvm::DenseMap<TraitApplicationAttr, FlatSymbolRefAttr> &memo,
                         PatternRewriter &rewriter);

} // end mlir::trait
