#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"
include "TypeInterfaces.td"

// Represents a universally quantified type parameter
def PolyType : TypeDef<TraitDialect, "Poly", [SymbolicTypeUnificationInterface]> {
  let mnemonic = "poly";
  let summary = "A polymorphic type parameter";

  let description = [{
    The `!trait.poly` type represents a universally quantified type parameter.
    It is used to model the parameters of generic functions.

    The `unique_id` serves as a unique identifier for this type within the module.

    Alternatively, `fresh` can be specified instead of `unique_id` to request a "fresh"
    `!trait.poly` with an automatically-generated `unique_id`. Fresh IDs are allocated in
    monotonically decreasing order beginning at `-1`.
  }];

  let extraClassDeclaration = [{
    // create a "fresh" PolyType with an unused unique_id
    static PolyType fresh(MLIRContext* ctx);

    // for SymbolicTypeUnificationInterface
    LogicalResult unifyWith(Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let parameters = (ins "int":$unique_id);

  let hasCustomAssemblyFormat = 1;
}

// Represents a zero-sized proof that a trait application is satisfied
def WitnessType : TypeDef<TraitDialect, "Witness"> {
  let mnemonic = "witness";
  let summary = "Proof that a type list implements a trait";

  let description = [{
    The `!trait.witness` type represents evidence that some list of types satisfies a trait.

    For example,

        !trait.witness<@PartialEq[i32, i32]>

    represents a proof that an implementation exists for `@PartialEq` with `i32` supplied for both type parameters.
  }];

  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$trait,  // referenced trait symbol
    ArrayRefParameter<"Type">:$type_args // ordered list of type arguments
  );

  let genVerifyDecl = 1;

  let assemblyFormat = "`<` $trait `[` $type_args `]` `>`";
}


#endif // TRAIT_TYPES_TD
