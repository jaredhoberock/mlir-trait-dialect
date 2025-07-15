#pragma once

#include <mlir/IR/BuiltinTypes.h>

namespace mlir::trait {

// forward declaration for SymbolicMatcherInterface
class TraitOp;

}

#include "TypeInterfaces.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"

namespace mlir::trait {

inline bool containsSymbolicType(Type ty) {
  bool found = false;
  ty.walk([&](Type sub) {
    if (isa<SymbolicTypeInterface>(sub))
      found = true;
  });
  return found;
}

inline Type applySubstitution(const llvm::DenseMap<Type,Type> &substitution,
                              Type ty) {
  // set up type replacer
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<Type> {
    auto it = substitution.find(t);
    return (it != substitution.end()) ? std::optional<Type>(it->second) : std::nullopt;
  });

  return replacer.replace(ty);
}

/// Attempt to unify `expectedTy` and `foundTy`.  If `expectedTy` or `foundTy` is
/// a PolyType, record/verify a substitution via unifyPolyType().  Otherwise,
/// if the type is composite (i.e. has immediate sub‐types), recurse into each
/// child.  If neither side is a PolyType or composite, require exact equality.
///
/// `substitution` maps each PolyType → the concrete Type chosen.  `moduleOp` is
/// used to look up TraitOps when resolving PolyType constraints.
LogicalResult unifyTypes(
    Type expectedTy,
    Type foundTy,
    ModuleOp moduleOp,
    llvm::DenseMap<Type,Type> &substitution,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics
LogicalResult unifyTypes(
    Type expectedTy,
    Type foundTy,
    ModuleOp moduleOp,
    llvm::DenseMap<Type,Type> &subst);

// As above, but discards diagnostics *and* the resulting substitution
LogicalResult unifyTypes(
    Type expectedTy,
    Type foundTy,
    ModuleOp moduleOp);

inline bool containsWitnessType(Type ty) {
  bool found = false;
  ty.walk([&](Type sub) {
    if (isa<WitnessType>(sub))
      found = true;
  });
  return found;
}

inline bool containsWitnessType(Attribute attr) {
  bool found = false;
  attr.walk([&](Attribute sub) {
    if (auto ta = dyn_cast<TypeAttr>(sub)) {
      if (containsWitnessType(ta.getValue()))
        found = true;
    }
  });
  return found;
}

inline bool opMentionsWitnessType(Operation *op) {
  // inspect operands
  for (Type t : op->getOperandTypes())
    if (containsWitnessType(t)) return true;

  // inspect result types
  for (Type t : op->getResultTypes())
    if (containsWitnessType(t)) return true;

  // inspect block arguments
  for (Region& r : op->getRegions())
    for (Block& b : r)
      for (Value arg : b.getArguments())
        if (containsWitnessType(arg.getType()))
          return true;

  // inspect attributes
  for (NamedAttribute attr : op->getAttrs())
    if (containsWitnessType(attr.getValue()))
      return true;

  return false;
}

} // end mlir::trait
