#ifndef TRAIT_TYPE_INTERFACES_TD
#define TRAIT_TYPE_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

// Marker interface for symbolic types that should not be lowered eagerly.
// Used by types like !trait.self & !trait.poly to indicate that they do not
// represent concrete types and require monomorphization.
def SymbolicTypeInterface : TypeInterface<"SymbolicTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Marker interface for types that represent symbolic placeholders. These types
    must not be monomorphized or lowered until they are substituted with a concrete
    type. Used to defer trait.impl lowering when self_type is symbolic.
  }];
}

def SymbolicMatcherInterface : TypeInterface<"SymbolicMatcherInterface", [SymbolicTypeInterface]> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    This interface extends SymbolicTypeInterface with a method `matches(Type ty)`
    that returns `true` if the given type satisfies the predicate represented by
    this symbolic type.

    It is intended to enable structural `trait.impl`s by allowing symbolic types
    to decide whether a concrete type matches their predicate.
  }];

  let methods = [
    InterfaceMethod<
      "Returns `true` if the given type satisfies the symbolic predicate",
      "bool", "matches",
      (ins "Type":$ty, "ModuleOp":$module)
    >
  ];
}

#endif // TRAIT_TYPE_INTERFACES_TD
