// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "TraitAttributes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>

namespace mlir { class PatternRewriter; }

namespace mlir::trait {

// Generated interface declarations below mention these types before the
// concrete helper classes are defined in this header.
class TraitOp;
class InstantiationMap;
class UnificationMap;
class SpecializationMap;
class ProjectionBindings;
class EvidenceBindings;
class CallSubstitution;
class ImplResolver;

}

#include "TraitTypeInterfaces.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TraitTypes.hpp.inc"

namespace mlir::trait {

int freshPolyTypeId();

inline Type applySubstitutionOnce(const llvm::DenseMap<Type,Type> &subst,
                                  Type root);
inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty);
inline void normalizeSubstitutionInPlace(llvm::DenseMap<Type,Type> &subst);

/// InstantiationMap: GenericTypeInterface -> UnificationTypeInterface.
///
/// Maps each generic type parameter encountered during instantiation to the
/// fresh unification variable allocated for that parameter. Reusing the mapping
/// preserves identity: repeated occurrences of the same generic instantiate to
/// the same fresh variable.
class InstantiationMap {
public:
  std::optional<UnificationTypeInterface> lookup(GenericTypeInterface key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(GenericTypeInterface key, UnificationTypeInterface value) {
    bindings[key] = value;
  }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  llvm::DenseMap<GenericTypeInterface, UnificationTypeInterface> bindings;
};

/// UnificationMap: UnificationTypeInterface -> Type.
///
/// Bindings accumulated while unifying two types. Keys are types that actively
/// participate in unification, such as inference variables and projections; the
/// values are the types they are known to equal.
class UnificationMap {
public:
  std::optional<Type> lookup(UnificationTypeInterface key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(UnificationTypeInterface key, Type value) { bindings[key] = value; }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  llvm::DenseMap<UnificationTypeInterface, Type> bindings;
};

/// SpecializationMap: GenericTypeInterface -> Type.
///
/// Concrete type arguments chosen for generic type parameters.
class SpecializationMap {
public:
  std::optional<Type> lookup(GenericTypeInterface key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(GenericTypeInterface key, Type value) {
    assert((!bindings.count(key) || bindings.lookup(key) == value) &&
           "specialization bindings must not be replaced with a different type");
    bindings[key] = value;
  }

  // A specialization is fully composed by construction, so one structural
  // substitution pass is enough.
  Type apply(Type ty) const { return applySubstitutionOnce(toTypeMap(), ty); }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

  static SpecializationMap fromTypeMap(const llvm::DenseMap<Type, Type> &subst) {
    SpecializationMap result;
    for (auto [key, value] : subst) {
      auto generic = dyn_cast<GenericTypeInterface>(key);
      assert(generic && "specialization keys must be generic types");
      result.bind(generic, value);
    }
    return result;
  }

private:
  friend class CallSubstitution;

  size_t bindingCount() const { return bindings.size(); }

  llvm::DenseMap<GenericTypeInterface, Type> bindings;
};

/// ProjectionBindings: ProjectionType -> Type.
///
/// Concrete associated type results for projection types.
class ProjectionBindings {
public:
  std::optional<Type> lookup(ProjectionType key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(ProjectionType key, Type value) {
    assert((!bindings.count(key) || bindings.lookup(key) == value) &&
           "projection bindings must not be replaced with a different type");
    bindings[key] = value;
  }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  friend class CallSubstitution;

  size_t bindingCount() const { return bindings.size(); }

  llvm::DenseMap<ProjectionType, Type> bindings;
};

/// EvidenceBindings: ClaimType -> ClaimType.
///
/// Maps unproven claim spellings to equivalent proven claim spellings discovered
/// while checking evidence.
class EvidenceBindings {
public:
  std::optional<ClaimType> lookup(ClaimType key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(ClaimType unproven, ClaimType proven) {
    assert(!unproven.isProven() && "evidence keys must be unproven claims");
    assert(proven.isProven() && "evidence values must be proven claims");
    assert((!bindings.count(unproven) || bindings.lookup(unproven) == proven) &&
           "evidence bindings must not be replaced with a different proof");
    bindings[unproven] = proven;
  }

  // Used by recursive proof verification to roll back an optimistic binding
  // when a nested obligation fails.
  void erase(ClaimType key) { bindings.erase(key); }

  bool empty() const { return bindings.empty(); }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result;
    for (auto [key, value] : bindings)
      result[key] = value;
    return result;
  }

private:
  friend class CallSubstitution;

  size_t bindingCount() const { return bindings.size(); }

  llvm::DenseMap<ClaimType, ClaimType> bindings;
};

/// ImplSpecialization: SpecializationMap + EvidenceBindings.
///
/// The complete set of type rewrites needed to specialize an impl method for a
/// proven self claim. Unlike CallSubstitution, this does not carry projection
/// bindings or require fixed-point closure.
class ImplSpecialization {
public:
  ImplSpecialization(SpecializationMap specialization,
                     EvidenceBindings evidenceBindings)
      : specialization(std::move(specialization)),
        evidenceBindings(std::move(evidenceBindings)) {}

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result = specialization.toTypeMap();
    for (auto [key, value] : evidenceBindings.toTypeMap())
      result[key] = value;
    return result;
  }

private:
  SpecializationMap specialization;
  EvidenceBindings evidenceBindings;
};

/// CallSubstitution: SpecializationMap + ProjectionBindings + EvidenceBindings.
///
/// The complete set of type rewrites needed to lower one call site.
class CallSubstitution {
public:
  explicit CallSubstitution(SpecializationMap specialization)
      : specialization(std::move(specialization)) {}

  const SpecializationMap &getSpecialization() const { return specialization; }
  EvidenceBindings &getEvidenceBindings() { return evidenceBindings; }

  // The components can expose bindings for one another, so call substitutions
  // must chase to a fixed point.
  Type apply(Type ty) const {
    return applySubstitutionToFixedPoint(toTypeMap(), ty);
  }

  llvm::DenseMap<Type, Type> toTypeMap() const {
    llvm::DenseMap<Type, Type> result = specialization.toTypeMap();
    for (auto [key, value] : projectionBindings.toTypeMap())
      result[key] = value;
    for (auto [key, value] : evidenceBindings.toTypeMap())
      result[key] = value;
    normalizeSubstitutionInPlace(result);
    return result;
  }
  LogicalResult close(TypeRange operandTypes, TypeRange resultTypes,
                      FunctionType formalTy, ModuleOp module,
                      ImplResolver &resolver, ::mlir::PatternRewriter &rewriter,
                      llvm::function_ref<InFlightDiagnostic()> err = nullptr);

private:
  void discoverProjectionBindings(TypeRange types, ImplResolver &resolver,
                                  ::mlir::PatternRewriter &rewriter);
  LogicalResult discoverEvidenceBindings(
      TypeRange types, ModuleOp module,
      llvm::function_ref<InFlightDiagnostic()> err = nullptr);

  size_t bindingCount() const {
    return specialization.bindingCount() + projectionBindings.bindingCount() +
           evidenceBindings.bindingCount();
  }

  SpecializationMap specialization;
  ProjectionBindings projectionBindings;
  EvidenceBindings evidenceBindings;
};

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
  // fast path: if the root itself is a PolymorphicTypeInterface,
  // call its predicate
  if (auto p = dyn_cast<PolymorphicTypeInterface>(root)) {
    return p.isPolymorphic();
  }

  // otherwise, just walk the type
  bool found = false;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance(); 

    if (auto p = dyn_cast<PolymorphicTypeInterface>(sub)) {
      if (p.isPolymorphic()) {
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

// A type is "ground" when it contains no PolymorphicTypeInterface nodes at all —
// no poly vars, no inference vars, no projections, no claims. Unlike
// isMonomorphicType, which asks whether any participant *reports* as polymorphic,
// isGroundType asks whether any participant *exists*. A monomorphic projection
// like !trait.proj<@Foo[i64], "Bar"> is monomorphic (no poly vars) but not
// ground (the projection still needs resolution).
inline bool isGroundType(Type root) {
  bool found = false;
  root.walk([&](Type sub) -> WalkResult {
    if (isa<PolymorphicTypeInterface>(sub)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !found;
}

/// Refine implementation for ops whose `inferReturnTypes` refuses to mint
/// fresh PolyTypes: when inference fails, there is nothing to refine and
/// verification accepts the declared result types as-is (an opaque
/// polymorphic input determines nothing about the result). When inference
/// succeeds, the default compatibility check applies. Ops declare
/// InferTypeOpInterface with ["refineReturnTypes"] and delegate here.
template <typename ConcreteOp>
LogicalResult refineUnlessUnmintable(MLIRContext *ctx,
                                     std::optional<Location> location,
                                     ValueRange operands, DictionaryAttr attrs,
                                     OpaqueProperties properties,
                                     RegionRange regions,
                                     SmallVectorImpl<Type> &returnTypes) {
  SmallVector<Type, 4> inferred;
  if (failed(ConcreteOp::inferReturnTypes(ctx, location, operands, attrs,
                                          properties, regions, inferred)))
    return success();
  if (!ConcreteOp::isCompatibleReturnTypes(inferred, returnTypes))
    return emitOptionalError(
        location, "'", ConcreteOp::getOperationName(), "' op inferred type(s) ",
        inferred, " are incompatible with return type(s) of operation ",
        returnTypes);
  return success();
}

// returns true iff every PolymorphicTypeInterface inside `root` is polymorphic,
// and at least one such participant exists
inline bool isPurelyPolymorphicType(Type root) {
  bool sawPoly = false;

  // fast path: if the root itself is a PolymorphicTypeInterface
  // call its predicate
  if (auto p = dyn_cast<PolymorphicTypeInterface>(root)) {
    if (p.isMonomorphic())
      return false; // root participates and is monomorphic -> not purely polymorphic
    sawPoly = true; // root participates and is polymorphic
  }

  // otherwise, walk the type and check every participating subtype
  bool allParticipatingArePoly = true;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance();

    if (auto p = dyn_cast<PolymorphicTypeInterface>(sub)) {
      if (p.isPolymorphic()) {
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

/// Instantiate a type with fresh inference variables.
///
/// For each GenericTypeInterface encountered in `t`, creates a fresh instance and
/// records the mapping in `inst`. The `idCounter` is used to generate unique IDs
/// for inference variables (should start at 0 for each instantiation context).
///
/// For structural types (e.g., FunctionType, TupleType), recursively instantiates
/// sub-elements and rebuilds the type.
///
/// For atomic types (e.g., i32, f64), returns the type unchanged.
///
/// This function is memoized via `inst` - if a GenericTypeInterface is encountered multiple
/// times within the same type structure, it maps to the same InferenceType.
Type instantiate(Type t, InstantiationMap& inst, uint64_t& idCounter);

inline void dumpSubstitution(const llvm::DenseMap<Type,Type> &subst) {
  for (auto [k,v] : subst) {
    llvm::errs() << k << " -> " << v << "\n";
  }
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

inline Type applySubstitutionOnce(const llvm::DenseMap<Type,Type> &subst,
                              Type root) {
  SpecializationMap specialization;
  for (auto [key, value] : subst)
    if (auto generic = dyn_cast<GenericTypeInterface>(key))
      specialization.bind(generic, value);

  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<std::pair<Type, WalkResult>> {
    if (auto generic = dyn_cast<GenericTypeInterface>(t)) {
      // GenericTypeInterface types own generic specialization entirely;
      // don't recurse into their result.
      return std::make_pair(generic.specializeWith(specialization), WalkResult::skip());
    }

    // Otherwise, check the full mixed map for non-generic bindings such as
    // projections and evidence claims.
    if (auto it = subst.find(t); it != subst.end()) {
      return std::make_pair(it->second, WalkResult::advance());
    }

    return std::nullopt;
  });
  return replacer.replace(root);
}

inline Type applySubstitutionToFixedPoint(const llvm::DenseMap<Type,Type> &subst,
                                          Type ty) {
  Type cur = ty;
  while (true) {
    Type next = applySubstitutionOnce(subst, cur);
    if (!next || next == cur) break;
    cur = next;
  }
  return cur;
}

/// Applies a GAT substitution: maps each type in `typeParams` to the
/// corresponding type in `assocTypeArgs`, then substitutes into `boundType`.
/// Returns the original `boundType` unchanged if `typeParams` is empty.
inline Type applyGATSubstitution(ArrayAttr typeParams,
                                 ArrayRef<Type> assocTypeArgs,
                                 Type boundType) {
  if (!typeParams || typeParams.empty())
    return boundType;
  assert(typeParams.size() == assocTypeArgs.size() &&
         "GAT arity mismatch: typeParams and assocTypeArgs must have the same size");
  DenseMap<Type,Type> gatSubst;
  for (auto [param, arg] : llvm::zip(typeParams, assocTypeArgs))
    gatSubst[cast<TypeAttr>(param).getValue()] = arg;
  return applySubstitutionToFixedPoint(gatSubst, boundType);
}

inline FailureOr<DenseMap<Type,Type>> composeSubstitutions(const DenseMap<Type,Type> &f,
                                                           const DenseMap<Type,Type> &g,
                                                           llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
  DenseMap<Type,Type> fog;

  for (const auto &[k, v] : f) {
    // rewrite v by g to a fixed point
    auto rewritten = applySubstitutionToFixedPoint(g, v);

    auto [it, inserted] = fog.try_emplace(k, rewritten);
    if (!inserted && it->second != rewritten) {
      if (err) err() << "conflicting substitution for " << k
                     << ": " << it->second << " vs " << rewritten;
      return failure();
    }
  }
  return fog;
}

/// Attempts to update `subst` so that the parameter type `formal`
/// is unified with the argument type `actual`.
///
/// This function applies the current substitution mapping to both `formal`
/// and `actual` before comparison. If the normalized types are identical,
/// the substitution is unchanged and the call succeeds.
///
/// Otherwise, `formal` is examined to determine how `actual` can serve as
/// its substitute:
///   - If `formal` implements `UnficationTypeInterface`, its
///     `unify` logic is invoked to extend `subst`.
///   - If `formal` and `actual` have the same type constructor and arity,
///     substitution recurses on their immediate subtypes.
///   - Otherwise, the types are considered incompatible and an error is
///     reported via `emitError`, if provided.
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    UnificationMap &subst);

/// As above, but discards the resulting substitution
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// As above, but discards diagnostics *and* the resulting substitution
LogicalResult unify(
    Type formal,
    Type actual,
    ModuleOp module);

/// Attempts to build a substitution which is the inverse of subst by mapping values in subst to keys
inline FailureOr<DenseMap<Type,Type>> invertSubstitution(
    const DenseMap<Type,Type> &subst,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
  DenseMap<Type,Type> inverted;
  for (const auto &[k, v] : subst) {
    auto [it, inserted] = inverted.try_emplace(v, k);
    if (!inserted && it->second != k) {
      if (err) err() << "substitution is not injective: conflicting inverse for "
                     << v << ": " << it->second << " vs " << k;
      return failure();
    }
  }
  return inverted;
}

/// Compute the substitution that specializes a possibly polymorphic `formal`
/// type so it unifies with an `actual` type.
///
/// This is the main helper for checking uses of polymorphic functions or values
/// against a concrete call site or expected signature:
///
///  * **Instantiation.** Replace every generic parameter found in both `formal`
///    and `actual` with fresh inference variables, so unification works even if
///    `actual` itself contains generics.
///  * **Unification.** Solve constraints so the instantiated `formal` and
///    instantiated `actual` become equal, producing a mapping from inference
///    variables to concrete types.
///  * **Back-projection.** Compose the inference solution back through the
///    instantiation map to yield a map from the original generics in `formal`
///    to fully resolved types. Any generics that came from `actual` remain as
///    generics; no inference variables remain.
///  * **Normalization.** Chase and collapse substitution chains so the map is
///    stable (no trivial self-maps, no stale inference variables).
///
/// The returned map always has keys that are the generic placeholders occurring
/// in `formal`. Values are “ground” relative to inference (no `!trait.infer`
/// left), though they may still mention generics if the `actual` side was also
/// generic.
///
/// Returns `failure()` if the two types cannot be unified. If `err` is supplied,
/// a diagnostic is emitted on failure.
FailureOr<SpecializationMap> buildSpecialization(
    Type formal,
    Type actual,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

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

// Collect distinct generic types `ty`
inline SmallVector<GenericTypeInterface,4> getGenericTypesIn(Type ty) {
  SmallVector<GenericTypeInterface, 4> result;
  DenseSet<Type> seen;

  auto collect = [&](Type ty) {
    if (auto varTy = dyn_cast<GenericTypeInterface>(ty)) {
      if (seen.insert(varTy).second) // first time we see it
        result.push_back(varTy);
    }
  };

  ty.walk(collect);
  return result;
}

/// Verify that a `proven` claim soundly proves the (possibly still polymorphic)
/// `unproven` claim and extend `subst` with a mapping when appropriate.
/// 
/// Notes:
/// - `unproven` may already have been normalized by earlier substitutions and
///   thus arrive already proven; if it matches `proven` we succeed immediately.
/// - Only records a mapping when converting an unproven form to its proven form;
///   no-op if `unproven == proven`.
/// - Recursively checks trait requirements and impl assumptions, ensuring all
///   subproofs are consistent and present.
LogicalResult verifyAndRecordProof(ClaimType unproven,
                                   ClaimType proven,
                                   ModuleOp module,
                                   EvidenceBindings &bindings,
                                   llvm::function_ref<InFlightDiagnostic()> err);

/// Walks the given type and records proven claim substitutions.
///
/// For every `ClaimType` node inside `ty` that carries a proof
/// (i.e. `isProven()`), this adds a mapping from its unproven form
/// (`claim.asUnproven()`) to the proven claim itself into `subst`.
/// If a conflicting mapping for the same unproven key already exists,
/// returns failure and emits an error through `err`.
LogicalResult recordProofBindingsIn(Type ty,
                                    ModuleOp module,
                                    EvidenceBindings &bindings,
                                    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

std::string generateMangledNameSuffixFor(TypeRange typeArgs);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const DenseMap<Type,Type> &subst,
    ArrayRef<GenericTypeInterface> typeParams);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams);

} // end mlir::trait
