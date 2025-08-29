#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Trait.td"
include "TraitAttributes.td"
include "TraitTypeInterfaces.td"

// Represents a polymorphic type variable
def PolyType : TypeDef<TraitDialect, "Poly", [MonomorphizableTypeInterface]> {
  let mnemonic = "poly";
  let summary = "A polymorphic type variable";

  let description = [{
    The `!trait.poly` type represents a *polymorphic type variable*.

    Semantics depend on its *position* in a substitution/unification.

    * Universal (formal-side): When `!trait.poly` appears on the *formal* side of
      `substituteWith` (e.g., in trait headers, impl/self applications, polymorphic
      function signatures, or method types defined by a trait), it is a *universally
      quantified* parameter. During unification, the callee binds the universal
      variable to the caller's actual type (subject to an occurs check).

    * Existential (actual-side): When `!trait.poly` appears on the *actual* side of
      `substituteWith` (e.g., a variable that occurs only in a `where` clause, or a
      caller-supplied placeholder for an associated type), it acts as an *existential
      witness*. The unifier may instantiate this actual-side variable to *whatever
      formal type* is required to satisfy the match (again with an occurs check).
      Practically, this is how "associated types" are named without introducing a new
      type former: the caller writes `!trait.poly` and the callee determines (and
      records) the concrete type via the substitution.

    * Normalization and consistency: All substitutions are applied eagerly via
      `applySubstitution`. If a variable is already bound, subsequent unifications
      must agree. Cycles are rejected (via an occurs check).

    * Polymorphic IR and monomorphization: `!trait.poly` is considered
      "purely polymorphic" and is allowed in polymorphic IR. After monomorphization,
      any remaining `!trait.poly` in SSA positions that require concrete shapes is a
      verifier error; callers that produce results using actual-side variables must
      have those variables resolved by the time the call is instantiated.

    Freshness and identity:
      The `unique_id` serves as the variable's identity within a module. Alternatively,
      `fresh` can be used to request a new `!trait.poly` with an automatically
      generated `unique_id` (allocated in monotonically decreasing order starting at
      `-1`).
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
