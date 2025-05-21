#pragma once

#include <mlir/IR/BuiltinTypes.h>

namespace mlir::trait {

// forward declaration for SymbolicMatcherInterface
class TraitOp;

}

#include "TypeInterfaces.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"
