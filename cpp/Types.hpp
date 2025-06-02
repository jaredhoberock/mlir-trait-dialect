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

/// Attempt to unify `expectedTy` and `foundTy`.  If `expectedTy` or `foundTy` is
/// a PolyType, record/verify a substitution via unifyPolyType().  Otherwise,
/// if the type is composite (i.e. has immediate sub‐types), recurse into each
/// child.  If neither side is a PolyType or composite, require exact equality.
///
/// `substitution` maps each PolyType → the concrete Type chosen.  `moduleOp` is
/// used to look up TraitOps when resolving PolyType constraints.
LogicalResult unifyTypes(
    Location loc,
    Type expectedTy,
    Type foundTy,
    ModuleOp moduleOp,
    llvm::DenseMap<Type,Type> &substitution);

}

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"
