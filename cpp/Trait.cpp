// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "LoweringContribution.hpp"
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"

#include <Trait.cpp.inc>

namespace mlir::trait {

namespace {
/// Monomorphization is the trait dialect's lowering step: it substitutes the
/// coordinate types the type system carried, so the step it contributes
/// discharges the coord dialect.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("monomorphize-trait", false, "", false);
    sink.dischargeDialect("coord");
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
