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
#include <mlir/Interfaces/FoldInterfaces.h>

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
void *instantiateLegality(MlirOperation, void *targetPtr, void *, void *) {
  auto &target = *static_cast<ConversionTarget *>(targetPtr);
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) -> std::optional<bool> { return !isPendingOp(op); });
  return nullptr;
}

/// The legality erase-polymorphs hands the driver: its readiness target
/// (`populateErasePolymorphsReadiness`), and a converter that maps every
/// polymorphic type to none. Erase is therefore present while a template stands,
/// or an op of this dialect, or a poly, claim, or projection type does, and runs
/// to cut the template; the readiness walk reads what erase collects off the same
/// interface the type system already speaks.
void *eraseLegality(MlirOperation, void *targetPtr, void *, void *) {
  auto &target = *static_cast<ConversionTarget *>(targetPtr);
  populateErasePolymorphsReadiness(target);
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
/// the monomorphs each trait call needs and proves the monomorphic application
/// claims and allegations, leaving the polymorphic templates standing; its target
/// has an opinion on every operation and marks a pending one illegal, so the
/// readiness walk runs it exactly while a pending operation stands. erase-polymorphs
/// then judges at its own entry every claim and projection standing outside a
/// template -- every equality among them, which nothing makes pending -- erases the
/// claims and projections, respells the remaining types, and collects the templates
/// nothing names; its target has no opinion on any operation still pending
/// instantiation, whatever its kind, so the readiness walk holds it behind
/// instantiate while one stands and the two never share a batch, and its converter
/// refuses every polymorphic type, so it runs once the type system has settled, and
/// it requests the cleanup interlude after it.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("instantiate-monomorphs", &addInstantiateMonomorphs,
                   &instantiateLegality, nullptr);
    sink.beginStep("erase-polymorphs", &addErasePolymorphs, &eraseLegality, nullptr,
                   /*wantsCleanup=*/true);
  }
};

/// A constant the folder materializes while simplifying a method's body is
/// placed in that body, as it is in an isolated function's.
///
/// The folder places a constant in the nearest enclosing region whose parent is
/// isolated from above or top-level, unless the dialect owning a parent on the
/// way claims that parent's region through this hook (`getInsertionRegion` in
/// upstream's `FoldUtils.cpp`). A method is not isolated from above, so without
/// the claim a method's constants would be pooled in its trait or impl, whose
/// child list refuses them; with it every constant stays in the method that
/// made it.
struct MethodConstantPlacement : DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const final {
    return isa<MethodOp>(region->getParentOp());
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

  addInterfaces<LoweringContribution, MethodConstantPlacement>();
}

}
