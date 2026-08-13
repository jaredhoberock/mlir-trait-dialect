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

/// Verifies an equality-armed projection-resolution `witness` against `module`;
/// passing an application-armed witness is a caller bug. Succeeds iff the cited
/// impl (`witness.getImplRef()`), specialized for the projection's application
/// and modulo the equality `premises`, binds the projection to the resolved
/// type, proofs ignored. The cited impl's own assumptions must each be covered
/// by the application-arm `obligationPremises` -- deliberately not its trait
/// requirements, which may quantify over GAT variables with no ground instance
/// here. This use-site entry resolves the actual side's ground projections by
/// module lookup. `err`, when non-null, receives the diagnostic on refusal.
LogicalResult verifyProjectionResolutionAtUse(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

/// The impl-birth companion to `verifyProjectionResolutionAtUse`, running the
/// same binding check and assumption discharge, differing in three ways. Its
/// head match is rigid -- only the cited impl's own generics instantiate -- so
/// the verdict is estate-independent. Its assumptions may also be covered by a
/// `dischargeWitnesses` entry, recursively over the same finite list. And on
/// success it returns the head-match substitution.
FailureOr<SpecializationMap> verifyProjectionResolutionAtBirth(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    ArrayRef<WitnessAttr> dischargeWitnesses,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

/// Rewrite a type with every proven application claim stripped to its unproven
/// form. Coerce comparison is modulo the proof, permanently.
Type stripClaimProofs(Type type);

/// The pending judgment a marked (unproven) coerce carries; one judgment serves
/// every checker of this evidence. The endpoints must unify, giving every
/// `!trait.proj` term a shared variable keyed by the projection itself: the same
/// projection is one variable, every other position is rigid, and a whole
/// projection is opaque (its arguments are not descended). A projection may
/// resolve to a projection-free position or to another bare projection (a direct
/// alias, both owed a grounding at discharge); a binding that resolves to a
/// composite still carrying a projection, or that closes a cycle, is refused.
/// Endpoints arrive with proofs already stripped. `err`, when non-null, receives
/// the diagnostic on refusal.
LogicalResult verifyPendingProjectionUnification(
    Type input, Type result,
    llvm::function_ref<InFlightDiagnostic()> emitError = nullptr);

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
