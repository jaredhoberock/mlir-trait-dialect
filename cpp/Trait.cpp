// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "LoweringContribution.hpp"
#include "NonFinalTypeInterface.hpp"
#include "Passes.hpp"
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>

#include <Trait.cpp.inc>

namespace mlir::trait {

namespace {
/// Add the instantiate-monomorphs pass to `pm`. It reads no per-invocation
/// options.
void addInstantiateMonomorphs(MlirOpPassManager pm, void *) {
  mlirOpPassManagerAddOwnedPass(pm, wrap(createInstantiateMonomorphsPass().release()));
}

/// Add the erase-polymorphs pass to `pm`. It reads no per-invocation options.
void addErasePolymorphs(MlirOpPassManager pm, void *) {
  mlirOpPassManagerAddOwnedPass(pm, wrap(createErasePolymorphsPass().release()));
}

/// Whether `op` outside a template is still pending instantiation: a generic call a
/// pattern would rewrite, or an operation carrying a standing obligation the pass must
/// reach. Both monomorphization steps read it. The instantiate step qualifies each of
/// its operation discharges by it, so a step is present exactly on the operations a
/// pattern fires on and an operation standing inside a template -- which this half
/// carries through untouched -- counts toward neither its presence nor its progress;
/// the erase step gates on it, ineligible while any such operation stands, so nothing
/// standing can still mention a template when it runs.
bool pendingOutsideTemplate(MlirOperation op, void *) {
  return isPendingOp(unwrap(op));
}

/// Monomorphization is the trait dialect's lowering, contributed as two steps so
/// another dialect's step may run between them without meeting a body this dialect
/// has already sealed. instantiate-monomorphs instantiates the monomorphs each
/// trait call needs and proves the monomorphic claims, leaving the polymorphic
/// templates standing; it discharges each trait call and each claim-producing
/// operation (allege, derive, project) qualified by pendingOutsideTemplate, so it is
/// present exactly on the operations a pattern fires on and leaves a template-interior
/// one for erase to take whole. Its verifier is on: with the monomorphs
/// instantiated and the polymorphic templates left standing, the module verifies
/// at the boundary between the two halves.
/// erase-polymorphs then erases the claims and projections resolved against those
/// templates, respells the remaining types, and collects the templates nothing
/// names; it discharges the coordinate types the type system carried and the
/// trait dialect's vocabulary -- all but the generic types standing inside nominal
/// attributes, which the nominal conversion takes with those attributes and which
/// the step therefore leaves for it. erase is ineligible while any op outside a
/// template is still pending instantiation, and it requests the cleanup interlude
/// that runs after it.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("instantiate-monomorphs");
    sink.passConstructor(&addInstantiateMonomorphs);
    sink.dischargeOperation("trait.func.call", &pendingOutsideTemplate, nullptr);
    sink.dischargeOperation("trait.method.call", &pendingOutsideTemplate, nullptr);
    sink.dischargeOperation("trait.allege", &pendingOutsideTemplate, nullptr);
    sink.dischargeOperation("trait.derive", &pendingOutsideTemplate, nullptr);
    sink.dischargeOperation("trait.project", &pendingOutsideTemplate, nullptr);
    sink.verifierPolicy(true);
    // monomorphization is the type system's own step: it runs while polymorphic
    // templates and unsettled claim, projection, and generic types stand, so it is
    // exempt from the non-final-type hold every conversion carries.
    sink.operatesOnNonFinalTypes();

    sink.beginStep("erase-polymorphs", /*wantsCleanup=*/true);
    sink.passConstructor(&addErasePolymorphs);
    sink.dischargeDialect("trait");
    sink.dischargeDialect("coord");
    sink.requiresAbsent(&pendingOutsideTemplate, nullptr);
    sink.operatesOnNonFinalTypes();
  }
};
} // namespace

// Each of the trait dialect's own types is a spelling the type system settles
// before the conversions run -- a polymorphic variable, a claim, a projection,
// or an inference variable -- so each declares its spelling not yet final under
// the family the driver names in the residual token and holds a conversion
// behind.
struct PolyNonFinal
    : public lowering::NonFinalTypeInterface::ExternalModel<PolyNonFinal, PolyType> {
  llvm::StringRef nonFinalFamily(Type) const { return "generic"; }
};
struct ClaimNonFinal
    : public lowering::NonFinalTypeInterface::ExternalModel<ClaimNonFinal, ClaimType> {
  llvm::StringRef nonFinalFamily(Type) const { return "claim"; }
};
struct ProjectionNonFinal
    : public lowering::NonFinalTypeInterface::ExternalModel<ProjectionNonFinal, ProjectionType> {
  llvm::StringRef nonFinalFamily(Type) const { return "projection"; }
};
struct InferenceNonFinal
    : public lowering::NonFinalTypeInterface::ExternalModel<InferenceNonFinal, InferenceType> {
  llvm::StringRef nonFinalFamily(Type) const { return "inference"; }
};

void TraitDialect::initialize() {
  registerAttributes();

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include <TraitOps.cpp.inc>
  >();

  addInterfaces<LoweringContribution>();

  PolyType::attachInterface<PolyNonFinal>(*getContext());
  ClaimType::attachInterface<ClaimNonFinal>(*getContext());
  ProjectionType::attachInterface<ProjectionNonFinal>(*getContext());
  InferenceType::attachInterface<InferenceNonFinal>(*getContext());
}

}
