#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"
include "TypeInterfaces.td"

// Represents the type of an impling type within a trait.method
def SelfType : TypeDef<TraitDialect, "Self", [SymbolicTypeInterface]> {
  let mnemonic = "self";
  let summary = "Placeholder for the implementing type within a trait or impl";

  let description = [{
    The `trait.self` type is a placeholder used inside `trait.method` and
    `trait.impl.method` to refer to the concrete type implementing the trait.

    This type is only valid within the context of trait declarations or implementations.
    It will be substituted with the actual concrete type during monomorphization.

    Example:
      trait.method @eq(%self: !trait.self, %other: !trait.self) -> i1
  }];

  let assemblyFormat = "";
}

// Represents a universally quantified type parameter constrained by a list of traits.
def PolyType : TypeDef<TraitDialect, "Poly", [SymbolicTypeInterface]> {
  let mnemonic = "poly";
  let summary = "A polymorphic type parameter constrained by one or more traits";

  let description = [{
    The `!trait.poly` type represents a universally quantified type parameter that is
    constrained by one or more traits. It is used to model generic functions with trait
    bounds, similar to Rust's generic type parameters like `T: Trait`.

    The `unique_id` serves as unique identifier for this type within the module. The `traits`
    field lists the required trait interfaces the type must satisfy, each given as a
    `SymbolRefAttr` referring to a `trait.trait` symbol.

    For example:
      !trait.poly<0, [@Add, @Clone]>
    refers to the first type parameter of a generic function, which must implement both
    the `Add` and `Clone` traits.
  }];

  let parameters = (ins
    "unsigned":$unique_id,
    OptionalArrayRefParameter<"::mlir::FlatSymbolRefAttr">:$traits
  );

  let assemblyFormat = "`<` $unique_id (`,` `[` $traits^ `]`)? `>`";
}

#endif // TRAIT_TYPES_TD
