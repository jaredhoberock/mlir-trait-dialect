// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Trait.td"
include "TraitAttributes.td"
include "TraitTypeInterfaces.td"

// Represents a polymorphic type variable
def PolyType : TypeDef<TraitDialect, "Poly", [
  GenericTypeInterface,
  PolymorphicTypeInterface
]> {
  let mnemonic = "poly";
  let summary = "A polymorphic type variable";

  let description = [{
    The `!trait.poly` type represents a *polymorphic type variable*.

    ### Position-dependent semantics

    Semantics depend on its *position* in a type unification.

    * Formal position (universally quantified):
      When a `!trait.poly` appears on the *formal* side of `unify` (e.g., in trait headers,
      impl/self applications, polymorphic function signatures, or trait-defined methods,
      it stands for a universally quantified type parameter. During unification, the callee
      may bind the universal variable to the caller's actual type, subject to an occurs check.

    * Actual position (existential):
      When a `!trait.poly` appears on the *actual* side of `unify` (e.g., in a caller's `where`
      clause or as an associated type placeholder), it acts as an existential variable.
      Unification may bind this existential to whatever the formal side demands.

    ### Interfaces

    `!trait.poly` implements:
    - `PolymorphicTypeInterface` - always reports itself as polymorphic.
    - `GenericTypeInterface` - supports instantiation into a fresh `InferenceType`
      when building an instantiation map.
    - `TypeVariableInterface` - can bind directly during unification when it appears on the *actual* side of unify.

    ### Identity
    
    Each `trait.poly` carries a unique integer `unique_id` to identify it within a module.
    `unique` can be used to request a new `!trait.poly` with an automatically
    generated `unique_id` (allocated in monotonically decreasing order starting at `-1`).
  }];

  let extraClassDeclaration = [{
    // create a PolyType with an unused unique_id
    static PolyType getUnique(MLIRContext* ctx);

    // for GenericTypeInterface
    Type instantiate(DenseMap<Type,Type> &inst, uint64_t &idCounter);

    // for PolymorphicTypeInterface
    bool isPolymorphic() const { return true; }
  }];

  let parameters = (ins "int":$unique_id);

  let hasCustomAssemblyFormat = 1;
}

// Represents a claim, possibly with direct proof, that a trait application is satisfied by some implementation
def ClaimType : TypeDef<TraitDialect, "Claim", [
  PolymorphicTypeInterface,
  UnificationTypeInterface
]> {
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

    bool isMonomorphic() const { return !isPolymorphic(); }

    bool isProven() const { return isMonomorphic() && static_cast<bool>(getProof()); }

    ClaimType asUnproven() const {
      return ClaimType::get(getContext(), getTraitApplication());
    }

    // bitmask flags for which projections to return from getProjections
    enum ProjectionKind {
      Identity              = 1 << 0,
      TraitRequirements     = 1 << 1,
      ProvenImplAssumptions = 1 << 2,
      AllProjections        = Identity | TraitRequirements | ProvenImplAssumptions,
    };

    /// Return the set of claims that can be legally projected from this claim.
    void getProjections(ModuleOp module,
                        SmallVectorImpl<ClaimType>& result,
                        ProjectionKind kinds = AllProjections);

    // for PolymorphicTypeInterface
    bool isPolymorphic() const;

    // for UnificationTypeInterface
    LogicalResult unify(Type other,
                        ModuleOp module,
                        DenseMap<Type,Type> &subst,
                        llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

// Represents an inference variable used during type inference
def InferenceType : TypeDef<TraitDialect, "Inference", [
  PolymorphicTypeInterface,
  TypeVariableInterface,
  UnificationTypeInterface
]> {
  let mnemonic = "infer";
  let summary = "INTERNAL: Type inference variable (should not appear in final IR)";

  let description = [{
    The `!trait.infer` type represents a *type inference variable* created during
    impl resolution and type checking. Each inference variable is created fresh when
    instantiating a polymorphic impl, ensuring that multiple uses of the same
    polymorphic impl get independent sets of type variables.

    Inference variables should NEVER appear in final IR - they are purely internal
    to the type checking process and should be eliminated by substitution before
    any IR is serialized or passed to subsequent compilation stages.
  }];

  let extraClassDeclaration = [{
    // for PolymorphicTypeInterface
    bool isPolymorphic() const { return true; }

    // for UnificationTypeInterface
    LogicalResult unify(Type other,
                        ModuleOp module,
                        DenseMap<Type,Type> &subst,
                        llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let parameters = (ins "uint64_t":$unique_id, "uint32_t":$origin_poly_id);

  let assemblyFormat = "`<` $unique_id `,` $origin_poly_id `>`";
}

#endif // TRAIT_TYPES_TD
