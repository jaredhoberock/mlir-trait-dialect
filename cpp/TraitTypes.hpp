#pragma once

#include "TraitAttributes.hpp"
#include <mlir/IR/BuiltinTypes.h>

namespace mlir::trait {

// forward declaration for SymbolicMatcherInterface
class TraitOp;

}

#include "TraitTypeInterfaces.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TraitTypes.hpp.inc"

namespace mlir::trait {

int freshPolyTypeId();

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

// returns true iff every MonomorphizableTypeInterface inside `root` is polymorphic,
// and at least one such participant exists
inline bool isPurelyPolymorphicType(Type root) {
  bool sawPoly = false;

  // fast path: if the root itself participates in monomorphization,
  // call its predicate
  if (auto m = dyn_cast<MonomorphizableTypeInterface>(root)) {
    if (m.isMonomorphic())
      return false; // root participates and is monomorphic -> not purely polymorphic
    sawPoly = true; // root participates and is polymorphic
  }

  // otherwise, walk the type and check every participating subtype
  bool allParticipatingArePoly = true;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance();

    if (auto m = dyn_cast<MonomorphizableTypeInterface>(sub)) {
      if (m.isPolymorphic()) {
        sawPoly = true;
        return WalkResult::advance();
      }
      // found a participant, monomorphic subtype -> fail
      allParticipatingArePoly = false;
      return WalkResult::interrupt();
    }

    return WalkResult::advance(); // non-participating types are ignored by design
  });

  // must have seen at least one one polymorphic participant, and none that are monomorphic
  return allParticipatingArePoly && sawPoly;
}

inline void normalizeSubstitutionInPlace(llvm::DenseMap<Type,Type> &subst) {
  // Snapshot keys so we can mutate the map safely.
  llvm::SmallVector<Type, 8> keys;
  keys.reserve(subst.size());
  for (auto &kv : subst) keys.push_back(kv.first);

  // Path-compressed chase with simple cycle guard + memo.
  llvm::DenseMap<Type, Type> memo;
  llvm::SmallPtrSet<Type, 8> inStack;

  auto chase = [&](Type t, auto &chase_ref) -> Type {
    // If t doesn’t map anywhere, it’s a fixed point.
    auto it = subst.find(t);
    if (it == subst.end()) return t;

    // Already memoized?
    if (auto mit = memo.find(t); mit != memo.end()) return mit->second;

    // Cycle guard: if we re-enter t, bail by treating t as fixed.
    if (!inStack.insert(t).second) return t;

    Type to = chase_ref(it->second, chase_ref);  // recurse
    memo[t] = to;                                // path compression
    inStack.erase(t);
    return to;
  };

  for (Type k : keys) {
    Type v = chase(k, chase);
    if (v == k) {
      // Drop trivial self-map.
      subst.erase(k);
    } else {
      subst[k] = v; // Collapse k directly to its fixed point.
    }
  }
}

inline llvm::DenseMap<Type,Type> normalizeSubstitution(llvm::DenseMap<Type,Type> subst) {
  normalizeSubstitutionInPlace(subst);
  return subst;
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

inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty) {
  Type cur = ty;
  while (true) {
    Type next = applySubstitution(subst, cur);
    if (!next || next == cur) break;
    cur = next;
  }
  return cur;
}
     

inline void dumpSubstitution(const llvm::DenseMap<Type,Type> &substitution) {
  for (auto [k,v] : substitution) {
    llvm::errs() << k << " -> " << v << "\n";
  }
}

/// Attempts to update `subst` so that the parameter type `formal`
/// is satisfied by the argument type `actual`.
///
/// This function applies the current substitution mapping to both `formal`
/// and `actual` before comparison. If the normalized types are identical,
/// the substitution is unchanged and the call succeeds.
///
/// Otherwise, `formal` is examined to determine how `actual` can serve as
/// its substitute:
///   - If `formal` implements `MonomorphizableTypeInterface`, its
///     `substituteWith` logic is invoked to extend `subst`.
///   - If `formal` and `actual` have the same type constructor and arity,
///     substitution recurses on their immediate subtypes.
///   - Otherwise, the types are considered incompatible and an error is
///     reported via `emitError`, if provided.
LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics
LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::DenseMap<Type,Type> &subst);

/// As above, but discards the resulting substitution
LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics *and* the resulting substitution
LogicalResult substituteWith(
    Type formal,
    Type actual,
    ModuleOp module);


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


inline SmallVector<PolyType,4> getPolyTypesIn(Type ty) {
  SmallVector<PolyType, 4> result;
  DenseSet<Type> seen;

  auto collect = [&](Type ty) {
    if (auto polyTy = dyn_cast<PolyType>(ty)) {
      if (seen.insert(polyTy).second) // first time we see it
        result.push_back(polyTy);
    }
  };

  ty.walk(collect);
  return result;
}

} // end mlir::trait
