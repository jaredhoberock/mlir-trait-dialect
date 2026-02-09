/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
