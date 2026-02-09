// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_TYPE_INTERFACES_TD
#define TRAIT_TYPE_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "Trait.td"
include "TraitAttributes.td"

// Interface for types that act as type parameters of generic declarations like
// polymorphic functions and traits
def GenericTypeInterface : TypeInterface<"GenericTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Types implementing this interface represent universally-quantified generics
    (e.g., `!trait.poly`) and participate in instantiation. They know how to
    instantiate themselves into a fresh type variable and memoize that choice
    into an instantiation map.
  }];

  let methods = [
    InterfaceMethod<
      "Instantiate `*this` to a fresh type variable (memoized in `inst`).",
      "mlir::Type", "instantiate",
      (ins "mlir::DenseMap<mlir::Type,mlir::Type>&":$inst,
           "uint64_t&":$idCounter)
    >,
    InterfaceMethod<
      "Specialize this type using the given substitution map. Returns the "
      "specialized type, or this type unchanged if no substitution applies. "
      "Wrapper types (e.g., `!coord.poly<N>`) should specialize "
      "their inner type and unwrap if the result is monomorphic.",
      "mlir::Type", "specializeWith",
      (ins "const mlir::DenseMap<mlir::Type,mlir::Type>&":$subst),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto it = subst.find($_type);
        return it != subst.end() ? it->second : $_type;
      }]
    >
  ];
}

// Interface for types that may contain polymorphic parts
def PolymorphicTypeInterface : TypeInterface<"PolymorphicTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Types implementing this interface report whether it is polymorphic itself,
    of (recursively) contains any subtype that is polymorphic. An instance is
    *monomorphic* iff it has no polymorphic parts. Monomorphic types can be
    lowered.
  }];

  let methods = [
    InterfaceMethod<
      "Returns true if this instance contains any polymorphic part.",
      "bool", "isPolymorphic"
    >
  ];

  let extraClassDeclaration = [{
    // convenience negates the result of isPolymorphic
    bool isMonomorphic() const {
      return !isPolymorphic();
    }
  }];
}

// Interface for types that actively participate in type unification
def UnificationTypeInterface : TypeInterface<"UnificationTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Types implementing this interface participate in type unification.
    An implementation defines how `*this` (the *formal* type on the left)
    unifies with an arbitrary right-hand side type, updating a caller-provided
    `DenseMap<Type,Type>` with *type-only* bindings.

    Guidelines:
    * Respect existing substitutions: implementations should assume the caller
      has normalized both sides via the current `subst` and may normalize again
      internally as needed.
    * Variable-like types (those modeling TypeVariableInterface) should perform
      an occurs check before binding to prevent infinite types.
  }];

  let methods = [
    InterfaceMethod<
      "Attempt to unify `*this` with `actual`, extending `subst`.",
      "llvm::LogicalResult", "unify",
      (ins "Type":$actual,
           "mlir::ModuleOp":$module,
           "mlir::DenseMap<mlir::Type,mlir::Type>&":$subst,
           "llvm::function_ref<mlir::InFlightDiagnostic()>":$emitError)
    >
  ];

  let extraClassDeclaration = [{
    // Convenience overload that discards diagnostics
    llvm::LogicalResult unify(
        Type actual,
        ModuleOp module,
        DenseMap<Type,Type> &subst) const {
      auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
      return this->unify(actual, module, subst, errFn);
    }
  }];
}

// Interface for types that behave like a type variable
// XXX TODO need to add generalize() method to map back to the original generic type
def TypeVariableInterface : TypeInterface<"TypeVariableInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Marker interface for *leaf, variable-like* types (e.g., `!trait.poly`,
    `!tuple.poly`). When such a type appears on the *actual* side of
    `unify(formal, actual, ...)`, it may bind *existentially* to the
    *formal* (with an occurs check), recording a subsitution `actual := formal`.
    Implementations need no methods; this is a pure marker.

    Types that implement `TypeVariableInterface` *must also implement* `UnificationTypeInterface`.
  }];
}

#endif // TRAIT_TYPE_INTERFACES_TD
