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

def SymbolicTypeUnificationInterface : TypeInterface<"SymbolicTypeUnificationInterface", [SymbolicTypeInterface]> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    This interface extends SymbolicTypeInterface with a method
    `unifyWith(Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError)`
    that returns `success` if the given type can be unified with this symbolic type.
    A `ModuleOp` is received as a parameter to allow symbol lookup.

    It is intended to enable structural `trait.impl`s by allowing symbolic types
    to decide whether a concrete type matches their predicate.
  }];

  let methods = [
    InterfaceMethod<
      "Returns `success` if the given type can be unified with this symbolic type",
      "llvm::LogicalResult", "unifyWith",
      (ins "Type":$ty, "ModuleOp":$module, "llvm::function_ref<InFlightDiagnostic()>":$emitError)
    >
  ];
}

#endif // TRAIT_TYPE_INTERFACES_TD
