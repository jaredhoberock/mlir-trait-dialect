#ifndef TRAIT_TYPE_INTERFACES_TD
#define TRAIT_TYPE_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "Attributes.td"
include "Dialect.td"

// Interface for types that may require monomorphization before lowering
def MonomorphizableTypeInterface : TypeInterface<"MonomorphizableTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Types implementing this interface participate in the monomorphization
    pipeline. An instance is *monomorphic* when it has no remaining symbolic
    parts and is ready to lower. Implementations may also own their unification
    against another type (recording substitutions as needed).
  }];

  let methods = [
    InterfaceMethod<
      "Returns true iff this instance is monomorphic (ready to lower).",
      "bool", "isMonomorphic"
    >,

    InterfaceMethod<
      "Unify `*this` with `other`, updating `substitution` as needed.",
      "llvm::LogicalResult", "unifyWith",
      (ins "Type":$other,
           "mlir::ModuleOp":$module,
           "mlir::DenseMap<mlir::Type,mlir::Type>&":$substitution,
           "llvm::function_ref<mlir::InFlightDiagnostic()>":$emitError)
    >
  ];

  let extraClassDeclaration = [{
    // silent unification, diagnostics are discarded
    LogicalResult unifyWith(
        Type other,
        ModuleOp module,
        DenseMap<Type,Type> &substitution) const {
      auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
      return this->unifyWith(other, module, substitution, errFn);
    }

    // convenience negates the result of isMonomorphic
    bool isPolymorphic() const {
      return !isMonomorphic();
    }
  }];
}

#endif // TRAIT_TYPE_INTERFACES_TD
