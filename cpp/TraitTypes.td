// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_TYPES_TD
#define TRAIT_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
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
    The `!trait.poly` type represents a *polymorphic type variable*: a type
    parameter some declaration binds, and the key a substitution over that
    declaration carries an argument under.

    ### Identity

    A `!trait.poly` carries a non-negative integer label. Labels are
    declaration-local: a declaration's parameters are read off its own header in
    position order, so a callee's `!trait.poly<0>` is its own first parameter
    and never the caller's, and two declarations may spell the same label
    without relating their parameters. A negative label is refused.

    ### Interfaces

    `!trait.poly` implements:
    - `PolymorphicTypeInterface` - always reports itself as polymorphic.
    - `GenericTypeInterface` - answers with the argument a substitution binds to
      the parameter it stands for.
  }];

  let extraClassDeclaration = [{
    // for GenericTypeInterface
    Type specializeWith(const ::mlir::trait::SpecializationMap &subst) const;

    // for PolymorphicTypeInterface
    bool isPolymorphic() const { return true; }
  }];

  let parameters = (ins "int":$label);

  let hasCustomAssemblyFormat = 1;
}

// Represents a claim, possibly with direct proof, that a trait application is satisfied by some implementation
def ClaimType : TypeDef<TraitDialect, "Claim", [
  PolymorphicTypeInterface,
  DeclareTypeInterfaceMethods<SymbolUserTypeInterface>
]> {
  let mnemonic = "claim";
  let summary = "Evidence that a predicate over types holds";

  let description = [{
    The `!trait.claim` type carries evidence that a *predicate* over types
    holds. Its first parameter is the predicate; there are two arms, distinct
    in meaning and sharing only the carrier:

    * `TraitApplicationAttr` — an implementation exists for a trait applied to
      a list of types. For example,

          !trait.claim<@PartialEq[i32,i32]>

      is an unproven claim that an implementation exists for `@PartialEq` with
      `i32` supplied for both type parameters, and

          !trait.claim<@PartialEq[i32,i32] by @PartialEq_impl_i32_i32>

      is a proven claim naming the impl (`@PartialEq_impl_i32_i32`) that
      establishes it.

    * `TypeEqualityAttr` — two type spellings name one type:

          !trait.claim<!A = !B>

      The equality arm never carries a proof; comparison on it is exact and
      direction-blind.

    Consumers dispatch on the arm; a claim of one arm is never acceptable where
    the other is required.

    A predicate attribute states a proposition. Constructing it does not imply
    the proposition holds — hypotheses and deliberately-false witness tests must
    be spellable. The evidence invariant lives one level up: a *value* of claim
    type, in verifier-valid IR, is evidence that its predicate holds. It is
    established by `trait.witness` for closed values and contractually assumed
    at function and block arguments.

    ### Why proofs are attached to the type

    Proof evidence is part of the type (rather than an attribute on
    the producing operation) so that any consumer can inspect it locally
    without dataflow or CFG analysis.  A claim value may flow through
    block arguments, function parameters, or other SSA indirections
    where the defining operation is not directly visible.

    ### Provenance doctrine (settled; do not relitigate)

    The IR is the sole authority on proof provenance.  A claim's proof
    is resolvable from the module text alone: from the proof spelled in
    the type, or in principle by searching the module's impls and
    introductions.  Every in-memory resolution structure (the record,
    the ledger, any memo) is an acceleration of that search and is
    never authoritative: no behavior may depend on memory state that
    the module text cannot reproduce.

    Removing spelled proofs from claim types in favor of side state
    was evaluated in depth and rejected by owner ruling: it makes the
    text depend on state outside itself.  Verifiers, by contrast,
    never require the spelled proof to agree with a re-derivation —
    comparisons across claim types are made modulo proof, so the
    spelling informs without ever being re-checked at a distance.
    Solving may consult the module and its caches; checking stays
    local.  Any future proposal to move provenance out of the spelled
    type requires an explicit owner ruling to reopen.
  }];

  let parameters = (ins
    "::mlir::Attribute":$predicate,
    "::mlir::FlatSymbolRefAttr":$proof
  );

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static ClaimType get(MLIRContext* ctx, TraitApplicationAttr app) {
      return ClaimType::get(ctx, app, /*proof=*/nullptr);
    }

    static ClaimType get(MLIRContext *ctx, FlatSymbolRefAttr trait, ArrayRef<Type> typeArgs) {
      return ClaimType::get(ctx, TraitApplicationAttr::get(ctx, trait, typeArgs));
    }

    // Build an equality-arm claim. The equality arm never carries a proof, so
    // the proof field is null by construction.
    static ClaimType getEquality(MLIRContext *ctx, TypeEqualityAttr eq) {
      return ClaimType::get(ctx, eq, /*proof=*/nullptr);
    }
    static ClaimType getEquality(MLIRContext *ctx, Type lhs, Type rhs) {
      return getEquality(ctx, TypeEqualityAttr::get(ctx, lhs, rhs));
    }

    // Arm predicates.
    bool isApplication() const {
      return static_cast<bool>(mlir::dyn_cast<TraitApplicationAttr>(getPredicate()));
    }
    bool isEquality() const {
      return static_cast<bool>(mlir::dyn_cast<TypeEqualityAttr>(getPredicate()));
    }

    // Arm-asserting accessor for the application arm; a caller that could also
    // meet an equality claim must dispatch on the arm first.
    TraitApplicationAttr getTraitApplication() const {
      auto app = mlir::dyn_cast<TraitApplicationAttr>(getPredicate());
      assert(app && "getTraitApplication() on a non-application claim; "
                    "the caller must dispatch on the claim arm");
      return app;
    }

    // Endpoint accessor for the equality arm; null on the application arm.
    TypeEqualityAttr getEqualityAttr() const {
      return mlir::dyn_cast<TypeEqualityAttr>(getPredicate());
    }

    bool isMonomorphic() const { return !isPolymorphic(); }

    bool isProven() const { return static_cast<bool>(getProof()); }

    ClaimType asUnproven() const {
      if (isEquality())
        return *this;
      return ClaimType::get(getContext(), getTraitApplication());
    }

    /// Return the set of claims that can be legally projected from this
    /// claim: the identity projection, the trait's requirements, and, when
    /// the claim is proven, its obligations spelled proven by the subproof
    /// discharging each and the impl's equality where-clauses.
    void getProjections(ModuleOp module,
                        SmallVectorImpl<ClaimType>& result);

    /// Whether this claim projects to `dst`: `dst` exactly matches one of the
    /// candidate projections `getProjections` enumerates. This is the one
    /// membership that defines a well-formed projection hop.
    bool projectsTo(ModuleOp module, ClaimType dst);

    // for PolymorphicTypeInterface
    bool isPolymorphic() const;
  }];

  let hasCustomAssemblyFormat = 1;
}

// Represents a projection of an associated type from a trait application
def ProjectionType : TypeDef<TraitDialect, "Projection", [
  PolymorphicTypeInterface,
  DeclareTypeInterfaceMethods<SymbolUserTypeInterface>
]> {
  let mnemonic = "proj";
  let summary = "A projected associated type from a trait application";

  let description = [{
    The `!trait.proj` type represents the projection of an associated type
    from a trait application. For example, `!trait.proj<@Iterator[!S], "Item">`
    represents `Iterator[!S]::Item`.

    This type is structural and survives into MLIR until monomorphization
    resolves it to a concrete type by consulting the matching `trait.impl`.
    Evidence (proof) is carried separately on `!trait.claim` operands, not
    on the projection type itself.
  }];

  let parameters = (ins
    "::mlir::trait::TraitApplicationAttr":$trait_application,
    "::mlir::StringAttr":$assoc_name,
    ArrayRefParameter<"Type">:$assoc_type_args
  );

  let extraClassDeclaration = [{
    // Construct an unproven ClaimType for this projection's trait application
    ClaimType asClaim() const {
      return ClaimType::get(getContext(), getTraitApplication());
    }

    // for PolymorphicTypeInterface
    bool isPolymorphic() const;
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_TYPES_TD
