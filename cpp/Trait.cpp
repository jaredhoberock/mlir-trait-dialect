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

/// The legality instantiate-monomorphs hands the driver: its readiness target
/// (`populateInstantiateMonomorphsReadiness`), on which every operation is legal
/// unless it is still pending instantiation, so instantiate is present exactly on
/// the operations a pattern would fire on and has an opinion on every other one.
/// It rewrites no type, so it hands back no converter.
void *instantiateLegality(MlirOperation, void *targetPtr, void *, void *) {
  populateInstantiateMonomorphsReadiness(
      *static_cast<ConversionTarget *>(targetPtr));
  return nullptr;
}

/// The legality erase-polymorphs hands the driver: its readiness target
/// (`populateErasePolymorphsReadiness`). Erase is therefore present while a
/// template stands, which it runs to cut, or an op it rewrites -- an evidence
/// op, or one mentioning a claim or a projection. No type keeps erasure from
/// being ready -- it refuses none -- so it hands back no converter.
void *eraseLegality(MlirOperation, void *targetPtr, void *, void *) {
  populateErasePolymorphsReadiness(*static_cast<ConversionTarget *>(targetPtr));
  return nullptr;
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
/// instantiate while one stands and the two never share a batch, and it requests
/// the cleanup interlude after it.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("instantiate-monomorphs", &addInstantiateMonomorphs,
                   &instantiateLegality, nullptr);
    sink.beginStep("erase-polymorphs", &addErasePolymorphs, &eraseLegality, nullptr,
                   /*wantsCleanup=*/true);
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
