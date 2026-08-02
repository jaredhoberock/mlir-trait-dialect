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
/// Monomorphization is the trait dialect's lowering, and it runs as two steps.
/// The first instantiates the monomorphs each trait call needs and leaves the
/// polymorphic templates standing, so it removes no whole class. The second
/// erases those templates and the claims and projections resolved against them;
/// the coordinate types the type system carried leave the module there, so that
/// step discharges the coord dialect.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("instantiate-monomorphs", false, "", false);
    sink.beginStep("erase-polymorphs", false, "", false);
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
