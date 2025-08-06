#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Attributes.td"
include "Dialect.td"
include "TypeInterfaces.td"

// Represents a universally quantified type parameter
def PolyType : TypeDef<TraitDialect, "Poly", [SymbolicTypeInterface]> {
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

    // for SymbolicTypeInterface
    LogicalResult isSubstitutableBy(Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let parameters = (ins "int":$unique_id);

  let hasCustomAssemblyFormat = 1;
}

// Represents a claim, possibly with direct proof, that a trait application is satisfied by some implementation
def ClaimType : TypeDef<TraitDialect, "Claim"> {
  let mnemonic = "claim";
  let summary = "Claim that a type list implements a trait";

  let description = [{
    The `!trait.claim` type represents evidence that some list of types satisfies a trait.

    For example,

        !trait.claim<@PartialEq[i32,i32]>

    represents an unproven claim that an implementation exists for `@PartialEq` with `i32` supplied for both type parameters.

    And
   
        !trait.claim<@PartialEq[i32,i32], @PartialEq_impl_i32_i32>

    represents a proven claim that an implementation (i.e, `@PartialEq_impl_i32_i32`) exists for `@PartialEq[i32,i32]`.
  }];

  let parameters = (ins
    "::mlir::trait::TraitApplicationAttr":$trait_application,
    "::mlir::FlatSymbolRefAttr":$proof
  );

  let extraClassDeclaration = [{
    static ClaimType get(MLIRContext* ctx, TraitApplicationAttr attr) {
      return ClaimType::get(ctx, attr, /*proof=*/nullptr);
    }

    static ClaimType get(MLIRContext *ctx, FlatSymbolRefAttr trait, ArrayRef<Type> typeArgs) {
      return ClaimType::get(ctx, TraitApplicationAttr::get(ctx, trait, typeArgs));
    }

    FlatSymbolRefAttr getTraitName() const {
      return getTraitApplication().getTrait();
    }

    ArrayRef<Type> getTypeArgs() const {
      return getTraitApplication().getTypeArgs();
    }

    bool isProven() const {
      return static_cast<bool>(getProof());
    }
  }];

  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_TYPES_TD
