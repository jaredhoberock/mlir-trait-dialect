#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Attributes.td"
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
def ProofType : TypeDef<TraitDialect, "Proof"> {
  let mnemonic = "proof";
  let summary = "Proof that a type list implements a trait";

  let description = [{
    The `!trait.proof` type represents evidence that some list of types satisfies a trait.

    For example,

        !trait.proof<@PartialEq[i32, i32]>

    represents a proof that an implementation exists for `@PartialEq` with `i32` supplied for both type parameters.
  }];

  let parameters = (ins
    "::mlir::trait::TraitApplicationAttr":$application
  );

  let extraClassDeclaration = [{
    static ProofType get(MLIRContext *ctx, FlatSymbolRefAttr trait, ArrayRef<Type> typeArgs) {
      return ProofType::get(ctx, TraitApplicationAttr::get(ctx, trait, typeArgs));
    }

    static ProofType get(MLIRContext *ctx, FlatSymbolRefAttr trait, ArrayAttr typesAttr) {
      return ProofType::get(ctx, TraitApplicationAttr::get(ctx, trait, typesAttr));
    }

    FlatSymbolRefAttr getTrait() const {
      return getApplication().getTrait();
    }

    ArrayRef<Type> getTypeArgs() const {
      return getApplication().getTypeArgs();
    }
  }];

  let genVerifyDecl = 1;

  let assemblyFormat = "`<` $application `>`";
}


#endif // TRAIT_TYPES_TD
