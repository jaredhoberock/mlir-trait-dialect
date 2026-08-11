// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include "TraitTypes.hpp"

namespace mlir::trait {

/// Mixin that adds a `getModule()` convenience method to any op.
template <typename ConcreteType>
class HasGetModule : public ::mlir::OpTrait::TraitBase<ConcreteType, HasGetModule> {
public:
  FailureOr<ModuleOp> getModule(
      llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
    auto module = this->getOperation()->template getParentOfType<ModuleOp>();
    if (!module) {
      if (err) err() << "not in a module";
      return failure();
    }
    return module;
  }
};

/// The projection-resolution certificate audit at the symbol seam, factored so
/// that both `WitnessOp::verifySymbolUses` and the C-API seam-audit query run
/// exactly this check on the same inputs. Looks the cited impl up in `module`,
/// resolves the `redex` projection through the impl's associated-type binding
/// specialized for the redex's trait application, applies the equality
/// `premises`, and compares the result against `contractum` receipt-blind.
/// Succeeds when the cited impl binds the redex to the contractum. `err` (never
/// null) receives the diagnostic on refusal; a caller that treats refusal as a
/// classification answer rather than an error suppresses it at the diagnostic
/// engine.
///
/// When `rigidHeadMatch` is set, the head match instantiates ONLY the cited
/// impl's own generics; the redex's application (the actual side) stays rigid, so
/// no module-visible impl resolves a projection spelled there and the audit
/// verdict never depends on the unrelated impls the module carries. The impl
/// birth audit sets it, so an impl's verdict is estate-independent. Left unset
/// (the default), the head match resolves the actual side's ground projections
/// by module lookup, as a witness-site audit does. When `outSubst` is non-null it
/// receives the head-match substitution, so a caller replaying the premise reuses
/// this build rather than deriving it a second time.
///
/// The audit additionally requires the cited impl's own assumptions --
/// specialized for the redex's application -- each to be discharged: either by a
/// hypothetical cover from the application-arm `obligationPremises` (the citing
/// impl's own where clause), or by a `dischargeCitations` entry whose spelled
/// application is the assumption and whose named impl, specialized for it, has
/// each of its own assumptions discharged in turn over the same finite citation
/// list. Both compare receipt-stripped and modulo the equality `premises`. It
/// deliberately does not reach the cited impl's trait requirements, which may
/// quantify over GAT variables with no ground instance at the witness;
/// requirement discharge belongs to the proof and birth machinery. So a witness
/// never cites an impl whose own assumptions are unmet.
LogicalResult auditProjResolveCertificate(
    ModuleOp module, Type redex, Type contractum, FlatSymbolRefAttr citedImpl,
    ArrayRef<TypeEqualityAttr> premises,
    llvm::function_ref<InFlightDiagnostic()> err,
    ArrayRef<TraitApplicationAttr> obligationPremises = {},
    ArrayRef<DischargeCitationAttr> dischargeCitations = {},
    bool rigidHeadMatch = false,
    SpecializationMap *outSubst = nullptr);

/// Rewrite a type with every proven application-claim receipt stripped to its
/// unproven form. Coerce comparison is modulo the receipt, permanently, so the
/// pending judgment and its consult run over receipt-stripped endpoints.
Type stripClaimReceipts(Type type);

/// The pending judgment a marked (unproven) coerce carries, factored so that
/// `CoerceOp::verify`, the instantiate lie-catch (which adds a birth-spelling
/// note), and the C-API consult all run exactly this check. The endpoints must
/// unify with every `!trait.proj` term a shared unification variable keyed by the
/// projection itself: the same projection is one variable, every other position
/// is rigid, and a whole projection is opaque (its arguments are not descended).
/// A projection may resolve to a projection-free position or to another bare
/// projection (a direct alias, both owed a grounding at discharge); a binding
/// that resolves to a composite still carrying a projection, or that closes a
/// cycle, is refused. Endpoints arrive with receipts already stripped. `err`
/// (never null) receives the diagnostic on refusal; a caller that treats refusal
/// as a classification answer suppresses it at the diagnostic engine.
LogicalResult verifyPendingProjectionUnification(
    Type input, Type result,
    llvm::function_ref<InFlightDiagnostic()> emitError);

} // end mlir::trait

namespace mlir::OpTrait {

template<class... ChildOps>
struct HasOnlyChildOps {
  template<class ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
  public:
    static LogicalResult verifyTrait(Operation* op) {
      for (auto &region : op->getRegions())
        for (auto &block : region)
          for (auto &child : block)
            if (!isa<ChildOps...>(child))
              return op->emitOpError() << "unexpected child op '"
                     << child.getName() << "'";
      return success();
    }
  };
};

} // end mlir::OpTrait


#define GET_OP_CLASSES
#include <TraitOps.hpp.inc>
