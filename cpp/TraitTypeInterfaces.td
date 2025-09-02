#ifndef TRAIT_TYPE_INTERFACES_TD
#define TRAIT_TYPE_INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "Trait.td"
include "TraitAttributes.td"

// Interface for types that may require monomorphization before lowering
def MonomorphizableTypeInterface : TypeInterface<"MonomorphizableTypeInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Types implementing this interface participate in the monomorphization
    pipeline. An instance is *monomorphic* when it has no remaining symbolic
    parts and is ready to lower. Implementations may also own their substitution
    with another type (recording substitutions as needed). Instances of this
    interface decide how they are substituted when they appear on the *formal*
    side of `substituteWith`.
  }];

  let methods = [
    InterfaceMethod<
      "Returns true iff this instance is monomorphic (ready to lower).",
      "bool", "isMonomorphic"
    >,

    InterfaceMethod<
      "Attempt to substitute `*this` with `actual`, updating `substitution` as needed.",
      "llvm::LogicalResult", "substituteWith",
      (ins "Type":$actual,
           "mlir::ModuleOp":$module,
           "mlir::DenseMap<mlir::Type,mlir::Type>&":$substitution,
           "llvm::function_ref<mlir::InFlightDiagnostic()>":$emitError)
    >
  ];

  let extraClassDeclaration = [{
    // silent substitution, diagnostics are discarded
    LogicalResult substituteWith(
        Type actual,
        ModuleOp module,
        DenseMap<Type,Type> &substitution) const {
      auto errFn = llvm::function_ref<InFlightDiagnostic()>{};
      return this->substituteWith(actual, module, substitution, errFn);
    }

    // convenience negates the result of isMonomorphic
    bool isPolymorphic() const {
      return !isMonomorphic();
    }
  }];
}

// Interface for types that behave like a type variable
def TypeVariableInterface : TypeInterface<"TypeVariableInterface"> {
  let cppNamespace = "::mlir::trait";
  let description = [{
    Marker interface for *leaf, variable-like* types (e.g., `!trait.poly`,
    `!tuple.poly`). When such a type appears on the *actual* side of
    `substituteWith(formal, actual, ...)`, it may bind *existentially* to the
    *formal* (with an occurs check), recording a subsitution `actual := formal`.
    Implementations need no methods; this is a pure marker.
  }];
}

#endif // TRAIT_TYPE_INTERFACES_TD
