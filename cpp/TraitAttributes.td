// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_ATTRIBUTES_TD
#define TRAIT_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "Trait.td"

def TraitApplicationAttr : AttrDef<TraitDialect, "TraitApplication",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "application";
  let summary = "A trait symbol applied to an ordered list of type arguments.";
  let description = [{
    Represents a trait symbol applied to an ordered list of type arguments.
    Used in places like `!trait.proof<@Trait[Args]>` or `where` clauses.

    Example:
      @PartialEq[i32,i32]
  }];

  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$trait_name, // name of referenced trait symbol
    ArrayRefParameter<"Type">:$type_args     // ordered list of type arguments
  );

  let extraClassDeclaration = [{
    static TraitApplicationAttr get(MLIRContext *ctx, FlatSymbolRefAttr traitName, ArrayAttr typesAttr) {
      SmallVector<Type, 4> types;
      types.reserve(typesAttr.size());
      for (TypeAttr ta : typesAttr.getAsRange<TypeAttr>()) {
        types.push_back(ta.getValue());
      }
      return TraitApplicationAttr::get(ctx, traitName, types);
    }

    // convenience converts from a StringRef for the trait name
    static TraitApplicationAttr get(MLIRContext *ctx, StringRef trait_name, ArrayRef<Type> types) {
      auto traitRef = FlatSymbolRefAttr::get(ctx, trait_name);
      return TraitApplicationAttr::get(ctx, traitRef, types);
    }

    FailureOr<TraitOp> getTrait(ModuleOp module, llvm::function_ref<InFlightDiagnostic()> errFn = nullptr) const;

    TraitOp getTraitOrAbort(ModuleOp module, const char* message) const;

    bool mentionsType(Type ty) const {
      bool found = false;
      for (Type arg : getTypeArgs()) {
        arg.walk([&](Type sub) {
          if (!found && sub == ty) found = true;
        });
        if (found) break;
      }
      return found;
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

// States that two type spellings name one type. The predicate carried by an
// equality-arm !trait.claim; see ClaimType.
def TypeEqualityAttr : AttrDef<TraitDialect, "TypeEquality", []> {
  let mnemonic = "equality";
  let summary = "A proposition that two type spellings name one type";
  let description = [{
    Represents the proposition that two types, `lhs` and `rhs`, denote one
    type. It is the equality-arm predicate of `!trait.claim`; the assembly form
    `!trait.claim<!A = !B>` carries it.

    Constructing this attribute states the proposition; it does not assert that
    the equality holds. Hypotheses and deliberately-false witness tests must be
    spellable, so the checked constructor enforces structural well-formedness
    only: the endpoints carry no proof receipts (a receipt would let the
    endpoint fall into the asymmetric proof comparison that the equality arm
    forbids). A *value* of the enclosing claim type, in verifier-valid IR, is
    the evidence that the equality holds.

    The endpoints are ordered; orientation is recorded for audit provenance
    only. Replay is direction-blind, so no symmetry construct exists.

    The endpoints are opaque to sub-element walking: the generic
    `AttrTypeReplacer` cannot visit or rewrite them, so they move only through
    the sanctioned clone rule (one substitution applied to both endpoints
    atomically, rebuilt through this constructor). Readers that must inspect an
    endpoint reach it through the dedicated endpoint accessors, never the walk.
  }];

  let parameters = (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs);

  // The endpoints live in a hand-written storage whose getAsKey() is
  // walk-opaque while the uniquing key retains both endpoints (see
  // TraitAttributes.cpp).
  let genStorageClass = 0;
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

// The frozen audit capsule of a projection-resolution equality witness. It
// records, immutably from birth, the equality the witness established and the
// impl that establishes it, so the witness verifier can replay the audit
// locally without a fresh module search.
def WitnessCertificateAttr : AttrDef<TraitDialect, "WitnessCertificate", []> {
  let mnemonic = "certificate";
  let summary = "The frozen audit of a projection-resolution equality witness";
  let description = [{
    Records a projection-resolution leaf: the equality `redex = contractum`
    holds because the cited `impl` binds the associated type named by `redex`
    (a projection carrying its member and any associated-type arguments) to
    `contractum`, once the impl is specialized for the projection's trait
    application.

    The relation is stored as a `TypeEqualityAttr`, so the frozen `redex` and
    `contractum` inherit that attribute's walk-opacity: no generic replacer
    reaches them, and readers use `getRedex`/`getContractum`, never the walk.
    The capsule's own storage is generated, so the endpoints stay opaque only
    through the equality's opacity -- the frozen capsules stay immutable only
    while no attribute-level replacement rule is registered for
    `TypeEqualityAttr`, since such a rule would rewrite every capsule's relation
    wholesale and break the instance-check and clone-rule contract. None is
    registered today; a future one must account for the capsules first.

    The cited impl, by contrast, is a plain symbol reference the generated
    storage leaves walk-visible, so its immutability is not structural: it rests
    on the estate registering no symbol-rename pass over this attribute (none
    exists today). A consistent rename would verify clean while rewriting frozen
    audit provenance in place; the only mover that preserves the audit is the
    clone rule's whole-capsule rebuild.
  }];

  let parameters = (ins
    "::mlir::trait::TypeEqualityAttr":$equality,
    "::mlir::FlatSymbolRefAttr":$cited_impl
  );

  let extraClassDeclaration = [{
    // The frozen relation's endpoints, read off the stored equality.
    Type getRedex() const { return getEquality().getLhs(); }
    Type getContractum() const { return getEquality().getRhs(); }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

// Names the impl that discharges one application obligation. An impl carries a
// flat list of these to discharge the standing assumptions of the conditional
// sibling impls its premises cite; see the description and
// auditProjResolveCertificate for the recursion.
def DischargeCitationAttr : AttrDef<TraitDialect, "DischargeCitation", []> {
  let mnemonic = "discharge";
  let summary = "Names the impl that discharges one application obligation";
  let description = [{
    Pairs an application obligation with the impl that discharges it. The impl
    verifier's premise audit uses these to discharge the assumptions a cited
    conditional impl leaves standing: an assumption is discharged either by a
    hypothetical cover from the citing impl's own where-clause applications, or
    by a discharge citation whose spelled `application` is that assumption and
    whose named `discharging_impl`, specialized for the application, has each of
    its own assumptions discharged in turn over this same finite list. The audit
    resolves each named impl by symbol and never scans the module for a
    discharger, so an impl's verdict does not depend on the estate around it.
  }];

  let parameters = (ins
    "::mlir::trait::TraitApplicationAttr":$application,
    "::mlir::FlatSymbolRefAttr":$discharging_impl
  );

  let hasCustomAssemblyFormat = 1;
}

// An ordered array of the one-hop predicates a trait requires -- its `where`
// clause; each entry is a trait application or a type equality (see the
// description).
def PredicateArrayAttr : AttrDef<TraitDialect, "PredicateArray",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "predicate_array";
  let summary = "An ordered array of predicates required by a trait";
  let description = [{
    Represents the ordered list of one-hop predicates a `trait.trait` requires
    (its `where` clause). Each entry is one of two arms, distinct in meaning:

      * a `TraitApplicationAttr` -- an implementation must exist for a trait
        applied to a list of types, e.g. `@Eq[!S]`;
      * a `TypeEqualityAttr` -- two spellings must name one type, e.g.
        `!trait.proj<@Trait[!S], "Assoc"> = i64`.

    Application entries are proved at impl selection; equality entries are
    checked at impl birth against the impl's own bindings. An application-only
    array prints and parses exactly as the trait application array it
    generalizes, so existing textual forms round-trip unchanged.

    Example:
      !S = trait.poly<0>
      trait.trait @FoldFn[!S] where [
        @Callable[!S],
        !trait.proj<@FoldFn[!S], "Output"> = !S
      ] { ... }
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Attribute">:$predicates
  );

  let extraClassDeclaration = [{
    // behave like a range of the raw predicate attributes
    auto begin() const { return getPredicates().begin(); }
    auto end() const { return getPredicates().end(); }
    bool empty() const { return getPredicates().empty(); }
    size_t size() const { return getPredicates().size(); }

    // The application-arm entries only, in declaration order. Consumers that
    // read a trait application off each requirement -- the proof/derive
    // obligation streams, acyclicity, unconditionality -- iterate these.
    llvm::SmallVector<TraitApplicationAttr> getApplications() const {
      llvm::SmallVector<TraitApplicationAttr> apps;
      for (::mlir::Attribute p : getPredicates())
        if (auto app = mlir::dyn_cast<TraitApplicationAttr>(p))
          apps.push_back(app);
      return apps;
    }

    // Whether any entry is an application requirement.
    bool hasApplications() const {
      return llvm::any_of(getPredicates(), [](::mlir::Attribute p) {
        return mlir::isa<TraitApplicationAttr>(p);
      });
    }

    // Convenience: build from application entries alone (the common case, and
    // the shape sibling dialects construct).
    static PredicateArrayAttr get(MLIRContext *ctx,
                                  ArrayRef<TraitApplicationAttr> apps) {
      SmallVector<Attribute> preds(apps.begin(), apps.end());
      return PredicateArrayAttr::get(ctx, preds);
    }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_ATTRIBUTES_TD
