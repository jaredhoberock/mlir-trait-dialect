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
/// Monomorphization is the trait dialect's lowering: one step spanning two
/// passes. The first instantiates the monomorphs each trait call needs and
/// leaves the polymorphic templates standing; the second erases those templates
/// and the claims and projections resolved against them. The step discharges
/// the coordinate types the type system carried and the trait dialect's
/// vocabulary — all but the generic types standing inside nominal attributes,
/// which the nominal conversion takes with those attributes and which the step
/// therefore leaves for it. The step requests the cleanup interlude that runs
/// after it, so it is that interlude's requester.
struct LoweringContribution : lowering::LoweringContributionInterface {
  using lowering::LoweringContributionInterface::LoweringContributionInterface;
  void contributeSteps(lowering::LoweringStepSink &sink) const override {
    sink.beginStep("monomorphize", /*wantsCleanup=*/true, "", false);
    sink.dischargeDialect("coord");
    sink.dischargeDialect("trait");
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
