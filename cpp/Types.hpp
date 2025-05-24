#pragma once

#include <mlir/IR/BuiltinTypes.h>

namespace mlir::trait {

// forward declaration for SymbolicMatcherInterface
class TraitOp;

}

#include "TypeInterfaces.hpp.inc"

namespace mlir::trait {

inline bool containsSymbolicType(Type ty) {
  // XXX TODO we actually need to traverse subelements of ty
  //          for this check to be correct
  return isa<SymbolicTypeInterface>(ty);
}

}

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"
