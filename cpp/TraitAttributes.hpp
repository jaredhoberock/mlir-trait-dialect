// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir::trait {
  // forward declaration of TraitOp for Attributes.td/Attributes.hpp.inc
  class TraitOp;
}

#define GET_ATTRDEF_CLASSES
#include "TraitAttributes.hpp.inc"

namespace mlir::trait {

inline Attribute applySubstitutionOnce(const llvm::DenseMap<Type,Type> &substitution,
                                   Attribute attr) {
  // set up type replacer
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<Type> {
    auto it = substitution.find(t);
    return (it != substitution.end()) ? std::optional<Type>(it->second) : std::nullopt;
  });

  return replacer.replace(attr);
}

} // end mlir::trait
