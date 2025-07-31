#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#define GET_ATTRDEF_CLASSES
#include "Attributes.hpp.inc"

namespace mlir::trait {

inline Attribute applySubstitution(const llvm::DenseMap<Type,Type> &substitution,
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
