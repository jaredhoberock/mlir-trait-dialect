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

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ImplResolution.hpp"
#include "Trait.hpp.inc"

namespace mlir::trait {

class ImplResolver;

struct MonomorphizationInterface : DialectInterface {
  inline MonomorphizationInterface(Dialect *dialect)
    : DialectInterface(dialect, TypeID::get<MonomorphizationInterface>())
  {}

  // Called during convert-to-traits
  // Register patterns that lower a dialect's operations into polymorphic trait IR
  //
  // If a dialect has rewrite patterns that can be run while operands are polymorphic,
  // then they should be registered by this method.
  //
  // Can emit any trait dialect operation.
  virtual void populateConvertToTraitPatterns(RewritePatternSet& patterns) const {}

  /// Called during instantiate-monomorphs
  /// Register patterns that *prepare and specialize* your dialect’s IR
  /// for monomorphization (e.g., concretize polymorphic region signatures,
  /// specialize helpers that carry !trait.claim values, etc.).
  ///
  /// If a dialect has rewrite patterns that cannot be run until operands are
  /// monomorphic, then they should be registered by this method.
  ///
  /// Constraints:
  ///   - MUST NOT introduce any of the following operations:
  ///     * trait.trait
  ///     * trait.impl
  ///     * trait.allege
  ///   - May assume proven claims / witnesses exist.
  virtual void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns) const = 0;

  /// Called during eraseClaims
  /// Provide a TypeConverter + rewrite patterns that erase all residual
  /// !trait.claim in your dialect’s types/regions and legalize ops after
  /// claims are dropped.
  virtual void populateEraseClaimsPatterns(TypeConverter &typeConverter, RewritePatternSet& patterns) const {}

  inline static StringRef getInterfaceName() { return "MonomorphizationInterface"; }

  inline static ::mlir::TypeID getInterfaceID() {
    return ::mlir::TypeID::get<MonomorphizationInterface>();
  }
};

class ImplGeneratorSet;

struct GenerateImplsInterface : DialectInterface {
  inline GenerateImplsInterface(Dialect *dialect)
    : DialectInterface(dialect, TypeID::get<GenerateImplsInterface>())
  {}

  virtual void populateImplGenerators(ImplGeneratorSet& generators) const = 0;

  inline static StringRef getInterfaceName() { return "GenerateImplsInterface"; }

  inline static TypeID getInterfaceID() {
    return TypeID::get<GenerateImplsInterface>();
  }
};

} // end mlir::trait
