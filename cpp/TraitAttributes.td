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

    The capsule is immutable from birth to death. Its `redex` and `contractum`
    are opaque to sub-element walking, so no generic replacer drifts them; the
    only sanctioned mover is the clone rule, which rebuilds the whole capsule
    through this constructor under one substitution. The witness verifier reads
    the fields through the accessors, never the walk.

    Renaming the cited impl (or a symbol inside a frozen endpoint) during the
    capsule's lifetime is forbidden; the sanctioned path is an audit-aware
    atomic rebuild.
  }];

  let parameters = (ins
    "::mlir::Type":$redex,
    "::mlir::Type":$contractum,
    "::mlir::FlatSymbolRefAttr":$cited_impl
  );

  // Hand-written storage (TraitAttributes.cpp) with no getAsKey, so the
  // type-bearing fields are opaque to walkers while uniquing retains them.
  let genStorageClass = 0;
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

// Names one application obligation and the impl that discharges it. An impl
// carries a flat list of these alongside its projection-resolution premises: a
// premise that cites a CONDITIONAL sibling impl leaves that impl's own
// assumptions to discharge, and where the citing impl's own where clause does
// not cover such an assumption, a discharge citation names the impl that does.
// The audit resolves the named impl by symbol and matches it against the
// obligation -- it never scans the module for a discharger. The obligation is
// spelled (`application`) so the audit compares against it directly; the
// discharger is named (`discharging_impl`) so a conditional discharger's own
// assumptions can in turn be discharged by further citations in the same list.
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
    its own assumptions discharged in turn over this same finite list.
  }];

  let parameters = (ins
    "::mlir::trait::TraitApplicationAttr":$application,
    "::mlir::FlatSymbolRefAttr":$discharging_impl
  );

  let hasCustomAssemblyFormat = 1;
}

def TraitApplicationArrayAttr : AttrDef<TraitDialect, "TraitApplicationArray",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "application_array";
  let summary = "An array of prerequisite trait applications required by traits or assumed by implementation";
  let description = [{
    Represents trait applications that are required to hold when implementing
    the enclosing trait. These assumptions are available as subproofs via
    `trait.project` on the self-proof obtained from `trait.assume`.

    This models a Rust-style `where` clause on a `trait` definition or implementation.
    It describes what must be assumed about type parameters in order for the trait’s methods
    to be implementable.

    Example:
      !S = trait.poly<0>
      trait.trait @Ord[!S] where [
        @Eq[!S],
        @PartialOrd[!S,!S]
      ]
      {
        ...
      }

    This indicates that implementing `@Ord[!S]` requires implementations of
    `@Eq[!S]` and `@PartialOrd[!S,!S]`.
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::trait::TraitApplicationAttr">:$applications
  );

  let extraClassDeclaration = [{
    auto getApplicationsOf(FlatSymbolRefAttr traitName) const {
      return llvm::make_filter_range(
        getApplications(),
        [traitName](TraitApplicationAttr app) {
          return app.getTraitName() == traitName;
        }
      );
    }

    // behave like a range of TraitApplicationAttr
    auto begin() const {
      return getApplications().begin();
    }

    auto end() const {
      return getApplications().end();
    }

    bool empty() const {
      return getApplications().empty();
    }

    size_t size() const {
      return getApplications().size();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

// An ordered array of the one-hop predicates that must hold whenever a trait
// holds -- a trait's `where` clause. It generalizes TraitApplicationArrayAttr
// with a second arm so a trait may require a type equality as well as a trait
// application.
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
    static PredicateArrayAttr get(MLIRContext *ctx,
                                  TraitApplicationArrayAttr apps) {
      SmallVector<Attribute> preds(apps.begin(), apps.end());
      return PredicateArrayAttr::get(ctx, preds);
    }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_ATTRIBUTES_TD
