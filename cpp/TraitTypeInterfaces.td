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
    (e.g., `!trait.poly`). A declaration binds them, a substitution keys them,
    and each answers for the argument its parameter takes.
  }];

  let methods = [
    InterfaceMethod<
      "Specialize this type using generic-type bindings. Returns the "
      "specialized type, or this type unchanged if no specialization applies. "
      "Wrapper types (e.g., `!coord.poly<N>`) should specialize their inner "
      "type and unwrap if the result is monomorphic.",
      "mlir::Type", "specializeWith",
      (ins "const ::mlir::trait::SpecializationMap&":$subst),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_type;
      }]
    >,
    InterfaceMethod<
      "The parameter this type is an occurrence of. A parameter is an "
      "occurrence of itself; a type that constrains the kind of a parameter "
      "(e.g. `!tuple.poly<!trait.poly<1>>`, which admits only a tuple) is an "
      "occurrence of the parameter it wraps, because a declaration binds the "
      "parameter and a substitution keys it.",
      "mlir::Type", "getParameterAtom",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_type;
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

#endif // TRAIT_TYPE_INTERFACES_TD
