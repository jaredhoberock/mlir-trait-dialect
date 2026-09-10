// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "LoweringContribution.hpp"
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

/// The legality instantiate-monomorphs hands the driver: every operation is legal
/// unless it is still pending instantiation, so instantiate is present exactly on
/// the operations a pattern would fire on and has an opinion on every other one.
/// It rewrites no type, so it hands back no converter.
void *instantiateLegality(MlirOperation, void *targetPtr, void *) {
  auto &target = *static_cast<ConversionTarget *>(targetPtr);
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) -> std::optional<bool> { return !isPendingOp(op); });
  return nullptr;
}

/// The legality erase-polymorphs hands the driver: the phase-1 target its pass
/// applies with a template marked illegal, and a converter that maps every
/// polymorphic type to none. Erase is therefore present while a template stands,
/// or a poly, claim, or projection type does, and runs to cut the template; the
/// readiness walk reads what erase collects off the same interface the type system
/// already speaks.
void *eraseLegality(MlirOperation, void *targetPtr, void *) {
  auto &target = *static_cast<ConversionTarget *>(targetPtr);
  populateErasePolymorphsLegality(target, /*templatesIllegal=*/true);
  auto *converter = new TypeConverter();
  converter->addConversion([](Type type) { return type; });
  converter->addConversion(
      [](Type type, SmallVectorImpl<Type> &) -> std::optional<LogicalResult> {
        if (isPolymorphicType(type))
          return success();
        return std::nullopt;
      });
  return converter;
}

/// Monomorphization is the trait dialect's lowering, contributed as two steps so
/// another dialect's step may run between them. instantiate-monomorphs instantiates
/// the monomorphs each trait call needs and proves the monomorphic claims, leaving
/// the polymorphic templates standing; its target has an opinion on every operation
/// and marks a pending one illegal, so the readiness walk runs it exactly while a
/// pending operation stands. erase-polymorphs then erases the resolved claims and
/// projections, respells the remaining types, and collects the templates nothing
/// names; its target holds it behind instantiate (it has no opinion on a pending
/// operation) and its converter refuses every polymorphic type, so it runs once the
/// type system has settled, and it requests the cleanup interlude after it.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("instantiate-monomorphs");
    sink.passConstructor(&addInstantiateMonomorphs);
    sink.verifierPolicy(true);
    sink.legality(&instantiateLegality, nullptr);

    sink.beginStep("erase-polymorphs", /*wantsCleanup=*/true);
    sink.passConstructor(&addErasePolymorphs);
    sink.legality(&eraseLegality, nullptr);
  }
};
} // namespace

void TraitDialect::initialize() {
  registerAttributes();

  registerTypes();

  addOperations<
#define GET_OP_LIST
#include <TraitOps.cpp.inc>
  >();

  addInterfaces<LoweringContribution>();
}

}
