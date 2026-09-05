// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Passes.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>
#include <mlir/Tools/Plugins/PassPlugin.h>

namespace mlir::trait {
namespace {

/// Reports the readiness predicates the instantiate and erase steps declare themselves by,
/// so a lit row can pin them directly rather than through their downstream
/// effect: a remark on each generic call carries isRewritableGenericCall, and a
/// remark on the module carries isPendingExpansion. Only the plugin registers
/// it; nothing the compiler builds runs it.
struct ReportExpansionReadinessPass
    : PassWrapper<ReportExpansionReadinessPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportExpansionReadinessPass);

  inline StringRef getArgument() const final { return "report-expansion-readiness-trait"; }
  inline StringRef getDescription() const final {
    return "Report isRewritableGenericCall per generic call and isPendingExpansion "
           "for the module through remarks.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (auto call = dyn_cast<FuncCallOp>(op))
        op->emitRemark() << "func.call @" << call.getCalleeNameAttr().getValue()
                         << " rewritable="
                         << (isRewritableGenericCall(op) ? "true" : "false")
                         << " foreign=" << (isForeign(op) ? "true" : "false");
      else if (isa<MethodCallOp>(op))
        op->emitRemark() << "method.call rewritable="
                         << (isRewritableGenericCall(op) ? "true" : "false")
                         << " foreign=" << (isForeign(op) ? "true" : "false");
    });
    module->emitRemark() << "pending-expansion="
                         << (isPendingExpansion(module) ? "true" : "false");
  }
};

} // namespace
} // namespace mlir::trait

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::trait::TraitDialect>();
  ::mlir::PassRegistration<::mlir::trait::VerifyAcyclicTraitsPass>();
  ::mlir::PassRegistration<::mlir::trait::ResolveImplsPass>();
  ::mlir::PassRegistration<::mlir::trait::InstantiateMonomorphsPass>();
  ::mlir::PassRegistration<::mlir::trait::ErasePolymorphsPass>();
  // Monomorphization as a whole is a pipeline, not a pass: instantiate the
  // monomorphs each trait call needs, then erase the residual polymorphism and
  // collect the templates nothing names. A registered pipeline name resolves
  // before a pass name, so a row spelling `monomorphize-trait` runs both.
  ::mlir::PassPipelineRegistration<>(
      "monomorphize-trait",
      "Instantiate monomorphs for trait calls, then erase all polymorphs and "
      "collect the templates nothing names.",
      [](::mlir::OpPassManager &pm) {
        pm.addPass(::mlir::trait::createInstantiateMonomorphsPass());
        pm.addPass(::mlir::trait::createErasePolymorphsPass());
      });
  // The freeze over the instantiation driver has nothing in a compilation that
  // asks it anything, so the pass that plants an ask is registered here and
  // nowhere the compiler builds from.
  ::mlir::PassRegistration<::mlir::trait::AskImplSelectionDuringInstantiationPass>();
  // Registered by the plugin alone: it reports the instantiate and erase steps' readiness
  // predicates for lit rows and has no place in a compilation.
  ::mlir::PassRegistration<::mlir::trait::ReportExpansionReadinessPass>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "TraitDialectPlugin",
    "v0.1",
    registerPlugin
  };
}
