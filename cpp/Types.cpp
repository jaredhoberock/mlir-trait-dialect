#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include "TypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}


/// Collect exactly the immediate child Types of `ty`. If `ty` has no sub‐elements,
/// returns an empty vector.
static SmallVector<Type, 4> getImmediateSubTypes(Type ty) {
  SmallVector<Type, 4> children;
  ty.walkImmediateSubElements(
      /*walkAttrsFn=*/[](Attribute) {
        // We don't need to collect Attributes here, so do nothing.
      },
      /*walkTypesFn=*/[&](Type subTy) {
        children.push_back(subTy);
      });
  return children;
}


static Type applySubstitution(
    const llvm::DenseMap<Type,Type> &substitution,
    Type ty) {
  if (auto it = substitution.find(ty); it != substitution.end()) {
    return it->second;
  }
  return ty;
}


/// Attempt to unify a PolyType `polyTy` against another type `otherTy`.
/// If `polyTy` already has a mapping in `substitution`, verify it matches
/// `otherTy`. Otherwise, ensure `otherTy` satisfies all trait requirements
/// of `polyTy` and record the mapping. To look up trait definitions, this
/// function walks up the IR from `loc` to find the enclosing ModuleOp.
static LogicalResult unifyPolyType(Location loc,
                                   PolyType polyTy,
                                   Type otherTy,
                                   ModuleOp moduleOp,
                                   llvm::DenseMap<Type, Type> &substitution) {
  // If we've already assigned a concrete type to this PolyType, check consistency.
  if (auto it = substitution.find(polyTy); it != substitution.end()) {
    if (it->second != otherTy) {
      return mlir::emitError(loc)
             << "mismatched substitution for poly type " 
             << polyTy.getUniqueId() << ": expected " 
             << it->second << ", found " << otherTy;
    }
    return success();
  }

  // Check that `otherTy` implements every trait listed on `polyTy`.
  for (auto traitAttr : polyTy.getTraits()) {
    auto traitRef = cast<FlatSymbolRefAttr>(traitAttr);
    // Use the symbol table to find the TraitOp definition.
    auto traitOp = dyn_cast_or_null<TraitOp>(
        SymbolTable::lookupSymbolIn(moduleOp, traitRef));
    if (!traitOp) {
      return mlir::emitError(loc)
             << "couldn't find trait '" << traitRef << "'";
    }
    // If `otherTy` does not implement this trait, error out.
    if (!traitOp.getImpl(otherTy)) {
      return mlir::emitError(loc)
             << "type " << otherTy 
             << " does not implement required trait " << traitRef 
             << " for poly type " << polyTy.getUniqueId();
    }
  }

  // Record the new substitution and succeed.
  substitution[polyTy] = otherTy;
  return success();
}


LogicalResult unifyTypes(Location loc,
                         Type expectedTy,
                         Type foundTy,
                         ModuleOp moduleOp,
                         llvm::DenseMap<Type, Type> &substitution) {
  // normalize both types by applying the current substitution
  expectedTy = applySubstitution(substitution, expectedTy);
  foundTy = applySubstitution(substitution, foundTy);

  // if the normalized types are equal, unification succeeds
  if (expectedTy == foundTy)
    return success();

  // expectedTy is a PolyType
  if (auto poly = dyn_cast<PolyType>(expectedTy)) {
    return unifyPolyType(loc,
                         poly,
                         foundTy,
                         moduleOp,
                         substitution);
  }

  // foundTy is a PolyType
  if (auto poly = dyn_cast<PolyType>(foundTy)) {
    return unifyPolyType(loc,
                         poly,
                         expectedTy,
                         moduleOp,
                         substitution);
  }

  // recurse into sub elements
  SmallVector<Type, 4> expectedKids = getImmediateSubTypes(expectedTy);
  SmallVector<Type, 4> foundKids    = getImmediateSubTypes(foundTy);

  // the number of subelements and TypeID of both types must match before recursing
  if (expectedKids.size() != foundKids.size() ||
      expectedTy.getTypeID() != foundTy.getTypeID()) {
    return emitError(loc)
           << "type mismatch: expected '" << expectedTy
           << "' but found '" << foundTy << "'";
  }

  // Recurse on each child pair
  for (unsigned i = 0, e = expectedKids.size(); i < e; ++i) {
    if (failed(unifyTypes(loc,
                          expectedKids[i],
                          foundKids[i],
                          moduleOp,
                          substitution)))
      return failure();
  }

  return success();
}


} // end mlir::trait
