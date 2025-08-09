#pragma once

#include "Attributes.hpp"
#include <mlir/IR/BuiltinTypes.h>

namespace mlir::trait {

// forward declaration for SymbolicMatcherInterface
class TraitOp;

}

#include "TypeInterfaces.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"

namespace mlir::trait {

// this walks a Type and looks for any occurrence of the given NeedleType
template<class NeedleType> bool containsType(Type ty) {
  bool found = false;
  ty.walk([&](Type sub) {
    if (isa<NeedleType>(sub))
      found = true;
  });
  return found;
}

inline bool isPolymorphicType(Type root) {
  // fast path: if the root itself participates in monomorphization,
  // call its predicate
  if (auto m = dyn_cast<MonomorphizableTypeInterface>(root)) {
    return m.isPolymorphic();
  }

  // otherwise, just walk the type
  bool found = false;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance(); 

    if (auto m = dyn_cast<MonomorphizableTypeInterface>(sub)) {
      if (m.isPolymorphic()) {
        found = true;
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return found;
}

inline bool isMonomorphicType(Type ty) {
  return !isPolymorphicType(ty);
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

inline void dumpSubstitution(const llvm::DenseMap<Type,Type> &substitution) {
  for (auto [k,v] : substitution) {
    llvm::errs() << k << " -> " << v << "\n";
  }
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

// this walks an Attribute and looks for any occurrence of the given NeedleType
template<class NeedleType> bool containsType(Attribute attr) {
  bool found = false;
  attr.walk([&](Attribute sub) {
    if (auto ta = dyn_cast<TypeAttr>(sub)) {
      if (containsType<NeedleType>(ta.getValue()))
        found = true;
    }
  });
  return found;
}

// this walks an Operation and looks for any occurrence of the given NeedleType
// note that this search does not recurse into child operations
template<class NeedleType> bool opMentionsType(Operation *op) {
  // inspect operands
  for (Type t : op->getOperandTypes())
    if (containsType<NeedleType>(t)) return true;

  // inspect result types
  for (Type t : op->getResultTypes())
    if (containsType<NeedleType>(t)) return true;

  // inspect block arguments
  for (Region& r : op->getRegions())
    for (Block& b : r)
      for (Value arg : b.getArguments())
        if (containsType<NeedleType>(arg.getType()))
          return true;

  // inspect attributes
  for (NamedAttribute attr : op->getAttrs())
    if (containsType<NeedleType>(attr.getValue()))
      return true;

  return false;
}

} // end mlir::trait
