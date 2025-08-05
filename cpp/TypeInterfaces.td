#ifndef TRAIT_TYPE_INTERFACES_TD
#define TRAIT_TYPE_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "Attributes.td"
include "Dialect.td"

// Interface for symbolic types to indicate that they do not represent concrete types and
// require monomorphization
def SymbolicTypeInterface : TypeInterface<"SymbolicTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Type interface for types that represent symbolic placeholders. These types
    must not be monomorphized or lowered until they are substituted with a concrete
    type.

    This interface exposes a method
    `isSubstitutableBy(Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError)`
    that returns `success` if this symbolic type can be substituted by the given type.
    A `ModuleOp` is received as a parameter to allow symbol lookup.
  }];

  let methods = [
    InterfaceMethod<
      "Returns `success` if this type can be substituted by the given type",
      "llvm::LogicalResult", "isSubstitutableBy",
      (ins "Type":$ty, "ModuleOp":$module, "llvm::function_ref<InFlightDiagnostic()>":$emitError)
    >
  ];

  let extraClassDeclaration = [{
    // silent substitution, diagnostics are discarded
    LogicalResult isSubstitutableBy(Type ty, ModuleOp module) const {
      auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
      return isSubstitutableBy(ty, module, errFn);
    }
  }];
}

#endif // TRAIT_TYPE_INTERFACES_TD
