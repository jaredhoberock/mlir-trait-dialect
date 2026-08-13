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
    only. A *value* of the enclosing claim type, in verifier-valid IR, is the
    evidence that the equality holds.

    Endpoints are statements of fact, so they admit no proven claims. A proven
    claim type glues a fact to one value's provenance (`by @impl`), and the
    provenance legitimately varies value by value while the fact does not. An
    equality must apply wherever its fact holds: if provenance could appear in
    an endpoint, either it would participate in matching — the same fact spelled
    `by @A` and `by @B` would look like two facts, correct coercions would be
    refused, and the composition judgment's endpoint matching would fail to
    connect valid chains — or it would be ignored everywhere, a dead field
    inviting misreading. The constructor forbids the state instead: one fact,
    one spelling. The single place value provenance meets fact-level evidence
    is the coerce verdict, which strips proofs at that one boundary.

    The endpoints are ordered; orientation is recorded for provenance
    only. Replay is direction-blind, so no symmetry construct exists.

    The endpoints are opaque to sub-element walking: the generic
    `AttrTypeReplacer` cannot visit or rewrite them. This carries the previous
    rule through the pipeline. Global type rewrites — proof stamping foremost —
    rewrite claim types wherever they can see them; a walkable endpoint would
    receive a stamped proof and become exactly the state the constructor
    forbids, so two individually-correct passes would kill a legal program.
    Sealing the endpoints puts that impossibility in the attribute itself
    rather than in an exclusion every present and future rewrite would have to
    remember. Endpoints move only through the sanctioned clone rule (one
    substitution applied to both endpoints atomically, rebuilt through this
    constructor, re-checked); readers reach them through the dedicated
    accessors, never the walk.
  }];

  let parameters = (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs);

  // Walk-opacity is a property of the storage: MLIR's walkers descend only
  // into storages exposing getAsKey(), and generated storage always exposes
  // it. Hand-writing the storage (no getAsKey; the uniquing key still retains
  // both endpoints) is the only way to seal the endpoints -- there is no
  // declarative opt-out. See TraitAttributes.cpp.
  let genStorageClass = 0;
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

// A predicate and the impl that witnesses it. One shape serves two roles, read
// off the predicate arm: an equality `projection = resolved` the impl's
// associated-type binding resolves (a projection-resolution witness), or a
// trait application the impl discharges (an obligation discharge).
def WitnessAttr : AttrDef<TraitDialect, "Witness", []> {
  let mnemonic = "witness";
  let summary = "A predicate and the impl that witnesses it";
  let description = [{
    Pairs a `predicate` with the `impl` that witnesses it -- an equality
    resolved by the impl's associated-type binding, or an application obligation
    the impl supplies. The predicate is one of two arms:

      * a `TypeEqualityAttr` `projection = resolved`, holding because the impl
        binds the associated type `projection` names (a projection carrying its
        member and any associated-type arguments) to `resolved`, once the impl
        is specialized for the projection's trait application.

      * a `TraitApplicationAttr`, an obligation the impl supplies, discharging a
        cited conditional sibling impl's standing assumptions; verification
        resolves the named impl by symbol and never scans the module, so an
        impl's verdict does not depend on the unrelated impls the module carries.

    Witnessing is one concept carried two ways, chosen by scope. This attribute
    IS the witness and stands alone as declaration-level witnessing. Where
    evidence must be a typed claim value, the `trait.witness` op wraps this same
    content and produces the value.

    An equality predicate keeps the walk-opacity of its own `TypeEqualityAttr`
    storage: no generic replacer reaches its endpoints, and readers use
    `getProjection`/`getResolved`, never the walk. An application predicate stays
    walk-visible. The witness's own storage is generated, so its immutability
    under substitution is not structural: it rests on no pass registering an
    attribute-level replacement rule for `TypeEqualityAttr` and no symbol-rename
    pass over this attribute (neither exists today). The only mover that
    preserves an immutable equality witness is the clone rule's rebuild of the
    whole witness: one substitution applied to both endpoints atomically, rebuilt
    through the checked constructor.
  }];

  let parameters = (ins
    "::mlir::Attribute":$predicate,
    "::mlir::FlatSymbolRefAttr":$impl_ref
  );

  let extraClassDeclaration = [{
    // The equality this witness resolves, read off the predicate; valid only
    // for an equality-headed witness.
    TypeEqualityAttr getEquality() const {
      return ::mlir::cast<TypeEqualityAttr>(getPredicate());
    }
    // The projection this witness resolves and the type it resolves to, read
    // off the equality endpoints; valid only for an equality-headed witness.
    Type getProjection() const { return getEquality().getLhs(); }
    Type getResolved() const { return getEquality().getRhs(); }
    // The application this witness discharges; valid only for an
    // application-headed witness.
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

    // The application-arm entries only, in declaration order.
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
