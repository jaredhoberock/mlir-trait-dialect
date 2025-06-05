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


LogicalResult PolyType::unifyWith(
    Type ty,
    ModuleOp module, 
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  // check that ty implements every trait listed in our trait bounds
  for (auto traitAttr : getTraits()) {
    auto traitRef = mlir::cast<FlatSymbolRefAttr>(traitAttr);
    // Use the symbol table to find the TraitOp definition.
    auto traitOp = mlir::dyn_cast_or_null<TraitOp>(
        SymbolTable::lookupSymbolIn(module, traitRef));
    if (!traitOp) {
      return emitError() << "couldn't find trait '" << traitRef << "'";
    }
    // If ty does not implement this trait, error out.
    if (!traitOp.getImpl(ty)) {
      return emitError()
             << "type " << ty 
             << " does not implement required trait " << traitRef 
             << " for poly type " << getUniqueId();
    }
  }

  return success();
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


/// Attempt to unify a SymbolicTypeUnificationInterface `symTy` against another type `otherTy`.
/// If `symTy` already has a mapping in `substitution`, verify it matches `otherTy`. Otherwise,
/// ensure `otherTy` can be unified with `symTy` and record the mapping.
static LogicalResult unifySymbolicType(Location loc,
                                       SymbolicTypeUnificationInterface symTy,
                                       Type otherTy,
                                       ModuleOp moduleOp,
                                       llvm::DenseMap<Type, Type> &substitution) {
  // If we've already substituted a concrete type for symTy, check consistency.
  if (auto it = substitution.find(symTy); it != substitution.end()) {
    if (it->second != otherTy) {
      return mlir::emitError(loc)
             << "mismatched substitution for type " 
             << symTy << ": expected " 
             << it->second << ", found " << otherTy;
    }
    return success();
  }

  // no substitution already exists, check that we can unify
  auto errFn = [loc] { return mlir::emitError(loc); };
  if (failed(symTy.unifyWith(otherTy, moduleOp, errFn))) {
    return failure();
  }

  // Record the new substitution and succeed.
  substitution[symTy] = otherTy;
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

  // expectedTy is a SymbolicTypeUnificationInterface
  if (auto symTy = dyn_cast<SymbolicTypeUnificationInterface>(expectedTy)) {
    return unifySymbolicType(loc,
                             symTy,
                             foundTy,
                             moduleOp,
                             substitution);
  }

  // foundTy is a SymbolicTypeUnificationInterface
  if (auto symTy = dyn_cast<SymbolicTypeUnificationInterface>(foundTy)) {
    return unifySymbolicType(loc,
                             symTy,
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
