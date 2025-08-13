#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Attributes.td"
include "Dialect.td"
include "TypeInterfaces.td"

// Represents a universally quantified type parameter
def PolyType : TypeDef<TraitDialect, "Poly", [MonomorphizableTypeInterface]> {
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

    // for MonomorphizableTypeInterface
    bool isMonomorphic() const {
      return false;
    }

    LogicalResult substituteWith(Type other,
                                 ModuleOp module,
                                 DenseMap<Type,Type> &substitution,
                                 llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let parameters = (ins "int":$unique_id);

  let hasCustomAssemblyFormat = 1;
}

// Represents a claim, possibly with direct proof, that a trait application is satisfied by some implementation
def ClaimType : TypeDef<TraitDialect, "Claim", [MonomorphizableTypeInterface]> {
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
    static ClaimType get(MLIRContext* ctx, TraitApplicationAttr app) {
      return ClaimType::get(ctx, app, /*proof=*/nullptr);
    }

    static ClaimType get(MLIRContext *ctx, FlatSymbolRefAttr trait, ArrayRef<Type> typeArgs) {
      return ClaimType::get(ctx, TraitApplicationAttr::get(ctx, trait, typeArgs));
    }

    LogicalResult verifySymbolUses(ModuleOp module, llvm::function_ref<InFlightDiagnostic()> err);

    bool isProven() const {
      return isMonomorphic() && static_cast<bool>(getProof());
    }

    // bitmask flags for which projections to return from getProjections
    enum ProjectionKind {
      Identity              = 1 << 0,
      TraitRequirements     = 1 << 1,
      ProvenImplAssumptions = 1 << 2,
      AllProjections        = Identity | TraitRequirements | ProvenImplAssumptions,
    };

    void getProjections(ModuleOp module,
                        SmallVectorImpl<ClaimType>& result,
                        ProjectionKind kinds = AllProjections);

    // for MonomorphizableTypeInterface
    bool isMonomorphic() const;
    LogicalResult substituteWith(Type other,
                                 ModuleOp module,
                                 DenseMap<Type,Type> &substitution,
                                 llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_TYPES_TD
