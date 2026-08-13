// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir::trait {
  // forward declaration of TraitOp for Attributes.td/Attributes.hpp.inc
  class TraitOp;

  // TypeEqualityAttr uses hand-written storage (defined in TraitAttributes.cpp)
  // so its endpoint types are opaque to sub-element walking; the generated
  // attribute class names the storage, so declare it first.
  namespace detail {
    struct TypeEqualityAttrStorage;
  }
}

#define GET_ATTRDEF_CLASSES
#include <TraitAttributes.hpp.inc>

namespace mlir { class AsmParser; }

namespace mlir::trait {

/// Parse the bracketed type-argument list of a trait application `@Trait[...]`
/// whose leading symbol `traitName` has already been read, and build the checked
/// application. This is the single grammar for the application body; the entry
/// token that precedes it -- a required symbol, or the optional symbol that
/// distinguishes an application from an equality in a where-clause predicate --
/// is the caller's to read. Fails, having emitted a diagnostic, on a malformed
/// argument list.
FailureOr<TraitApplicationAttr>
parseTraitApplicationBody(AsmParser &parser, FlatSymbolRefAttr traitName);

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
