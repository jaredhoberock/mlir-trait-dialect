// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Passes.hpp"
#include "Trait.hpp"
#include <mlir/Pass/PassManager.h>
#include <mlir/Tools/Plugins/DialectPlugin.h>
#include <mlir/Tools/Plugins/PassPlugin.h>

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::trait::TraitDialect>();
  ::mlir::PassRegistration<::mlir::trait::ConvertToTraitPass>();
  ::mlir::PassRegistration<::mlir::trait::VerifyMonomorphsPass>();
  ::mlir::PassRegistration<::mlir::trait::VerifyAcyclicTraitsPass>();
  ::mlir::PassRegistration<::mlir::trait::ResolveImplsPass>();
  ::mlir::PassRegistration<::mlir::trait::InstantiateMonomorphsPass>();
  ::mlir::PassRegistration<::mlir::trait::MonomorphizePass>();
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
