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
    only: no endpoint contains a proven claim (a proven claim would let the
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

// A predicate and the impl that witnesses it. One shape serves two roles, read
// off the predicate arm: an equality `redex = contractum` the impl's
// associated-type binding resolves (the projection-resolution certificate an
// impl's `premises` array and a `trait.witness` certificate slot carry), or a
// trait application the impl discharges (the obligation-discharge citation an
// impl's `discharges` array carries).
def WitnessAttr : AttrDef<TraitDialect, "Witness", []> {
  let mnemonic = "witness";
  let summary = "A predicate and the impl that witnesses it";
  let description = [{
    Pairs a `predicate` with the `impl` that witnesses it -- an equality
    resolved by the impl's associated-type binding, or an application obligation
    the impl supplies. The predicate is one of two arms:

      * a `TypeEqualityAttr` `redex = contractum`, resolved because the impl
        binds the associated type named by `redex` (a projection carrying its
        member and any associated-type arguments) to `contractum`, once the impl
        is specialized for the projection's trait application. This is the leaf
        an impl's `premises` array and a `trait.witness` certificate slot carry.

      * a `TraitApplicationAttr`, an obligation the impl supplies. This is the
        discharge an impl's `discharges` array carries to cover a cited
        conditional sibling impl's standing assumptions; the audit resolves the
        named impl by symbol and never scans the module, so an impl's verdict
        does not depend on the estate around it.

    An equality predicate keeps the walk-opacity of its own `TypeEqualityAttr`
    storage: no generic replacer reaches its endpoints, and readers use
    `getRedex`/`getContractum`, never the walk. An application predicate stays
    walk-visible. The witness's own storage is generated, so its immutability
    under substitution is not structural: it rests on the estate registering no
    attribute-level replacement rule for `TypeEqualityAttr` and no symbol-rename
    pass over this attribute (neither exists today). The only mover that
    preserves a frozen equality witness is the clone rule's whole-capsule
    rebuild.
  }];

  let parameters = (ins
    "::mlir::Attribute":$predicate,
    "::mlir::FlatSymbolRefAttr":$impl_ref
  );

  let extraClassDeclaration = [{
    // The equality this citation resolves, read off the predicate; valid only
    // for an equality-headed citation.
    TypeEqualityAttr getEquality() const {
      return ::mlir::cast<TypeEqualityAttr>(getPredicate());
    }
    Type getRedex() const { return getEquality().getLhs(); }
    Type getContractum() const { return getEquality().getRhs(); }
    // The application this citation discharges; valid only for an
    // application-headed citation.
    TraitApplicationAttr getApplication() const {
      return ::mlir::cast<TraitApplicationAttr>(getPredicate());
    }
  }];

  let genVerifyDecl = 1;
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
