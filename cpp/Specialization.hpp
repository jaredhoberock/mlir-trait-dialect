// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "TraitOps.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>

namespace mlir::trait {

/// Builds a type replacer that chases each stamped type to the substitution's
/// fixed point. When `module` is non-null it also resolves the ground
/// projections the substitution mints (a concrete argument substituted
/// into a projection spelling) by module-visible impl lookup, so a specialized
/// monomorph carries no ground projection that a unique module-visible impl
/// resolves; generator-pending and multi-candidate ground projections survive
/// unchanged. A null `module` performs no such lookup.
AttrTypeReplacer makeTypeReplacerFromSubstitution(const DenseMap<Type,Type> &subst,
                                                  ModuleOp module);

func::FuncOp specializePolymorph(OpBuilder& builder,
                                 func::FuncOp polymorph,
                                 StringRef instanceName,
                                 const DenseMap<Type,Type> &substitution);

void specializePolymorphicRegion(OpBuilder& builder,
                                 Region& polymorph,
                                 Region& monomorph,
                                 const DenseMap<Type,Type> &substitution);

}
