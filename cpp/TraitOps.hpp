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

/// Verifies a projection-resolution witness against `module`. The
/// equality-armed `witness` supplies it: `getProjection()` and
/// `getResolved()` name the projection and the type it resolves
/// to, cited to `witness.getImplRef()`. Passing an application-armed witness is
/// a caller bug. Looks the cited impl up in `module`, resolves the projection
/// through the impl's associated-type binding specialized for the projection's
/// trait application, applies the equality `premises`, and compares the result
/// against the resolved type proof-blind. Succeeds when the cited impl binds the
/// projection to the resolved type. `err`, when non-null, receives the
/// diagnostic on refusal.
///
/// Verification additionally requires the cited impl's own assumptions --
/// specialized for the projection's application -- each to be discharged by a
/// hypothetical cover from the application-arm `obligationPremises` (the citing
/// impl's own where clause), compared proof-stripped and modulo the equality
/// `premises`. It deliberately does not reach the cited impl's trait
/// requirements, which may quantify over GAT variables with no ground instance
/// at the witness; requirement discharge belongs to the proof and birth
/// machinery. So a witness never cites an impl whose own assumptions are unmet.
///
/// This use-site entry resolves the actual side's ground projections by module
/// lookup, as verifying a witness at its use site does, and admits no discharge
/// citations.
LogicalResult verifyProjectionResolutionAtUse(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    llvm::function_ref<InFlightDiagnostic()> err);

/// The impl-birth companion to `verifyProjectionResolutionAtUse`, running the
/// same binding check and assumption discharge over the shared core and
/// differing in three ways. Its head match is rigid: it instantiates ONLY the
/// cited impl's own generics, so the projection's application (the actual side)
/// stays rigid and no module-visible impl resolves a projection spelled there --
/// an impl's verdict cannot then turn on the unrelated impls the module carries.
/// Its assumptions may be discharged not only by `obligationPremises` but by a
/// `dischargeWitnesses` entry whose spelled application is the assumption and
/// whose named impl, specialized for it, has each of its own assumptions
/// discharged in turn over the same finite citation list. And on success it
/// returns the head-match substitution.
FailureOr<SpecializationMap> verifyProjectionResolutionAtBirth(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    ArrayRef<WitnessAttr> dischargeWitnesses,
    llvm::function_ref<InFlightDiagnostic()> err);

/// Rewrite a type with every proven application claim stripped to its unproven
/// form. Coerce comparison is modulo the proof, permanently.
Type stripClaimProofs(Type type);

/// The pending judgment a marked (unproven) coerce carries; one judgment serves
/// every checker of this evidence. The endpoints must
/// unify with every `!trait.proj` term a shared unification variable keyed by the
/// projection itself: the same projection is one variable, every other position
/// is rigid, and a whole projection is opaque (its arguments are not descended).
/// A projection may resolve to a projection-free position or to another bare
/// projection (a direct alias, both owed a grounding at discharge); a binding
/// that resolves to a composite still carrying a projection, or that closes a
/// cycle, is refused. Endpoints arrive with proofs already stripped. `err`, when
/// non-null, receives the diagnostic on refusal.
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
