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

    The endpoints are ordinary sub-elements: every walk reaches them, so a
    symbol sealed in one is reached by the framework's own symbol-user driver
    and by any walk that classifies content. What no rewrite may do is move
    one. Global type rewrites — proof stamping foremost — rewrite claim types
    wherever they can see them; an endpoint that received a stamped proof would
    be exactly the state the constructor forbids, so two individually-correct
    passes would kill a legal program. Every `AttrTypeReplacer` the dialect
    builds therefore carries one attribute-level rule that returns this
    attribute unchanged and skips its interior (`makeEndpointSealedReplacer`,
    TraitTypes.hpp): endpoints are a leaf to a rewrite and reachable to a read.
    The one sanctioned mover is the clone rule — one substitution applied to
    both endpoints atomically, rebuilt through this constructor, re-checked.
  }];

  let parameters = (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs);

  let genVerifyDecl = 1;

  // Prints `lhs = rhs`. The leading empty literal drops the space the
  // declarative printer emits before a leading parameter, so an equality
  // printed directly into an enclosing `<...>` reads with no gap after the `<`.
  let assemblyFormat = "`` $lhs `=` $rhs";
}

// One entry of a substitution: a type parameter a declaration binds, and the
// argument it takes.
def TypeBindingAttr : AttrDef<TraitDialect, "TypeBinding", []> {
  let mnemonic = "binding";
  let summary = "A declaration's type parameter and the argument it takes";
  let description = [{
    Pairs a type `parameter` -- one a declaration binds, spelled as that
    declaration spells it -- with the `argument` it takes. A list of these is
    a substitution keyed by the declaration's own parameters, so the entries
    carry no order a reader must agree on. The parameter is the label itself,
    never a wrapper constraining one.
  }];

  let parameters = (ins "::mlir::Type":$parameter, "::mlir::Type":$argument);

  let genVerifyDecl = 1;

  // Prints `parameter = argument`; see `TypeEqualityAttr` for the leading
  // empty literal.
  let assemblyFormat = "`` $parameter `=` $argument";
}

// A predicate and the impl that witnesses it. One shape serves two roles, read
// off the predicate arm: an equality `projection = resolved` the associated-type
// binding of the impl, at the arguments the witness carries for its parameters,
// resolves (a projection-resolution witness), or a trait application the impl
// discharges (an obligation discharge).
def WitnessAttr : AttrDef<TraitDialect, "Witness",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "witness";
  let summary = "A predicate and the impl that witnesses it";
  let description = [{
    Pairs a `predicate` with the `impl` that witnesses it -- an equality
    resolved by the impl's associated-type binding, or an application obligation
    the impl supplies. The predicate is one of two arms:

      * a `TypeEqualityAttr` `projection = resolved`, holding because the impl,
        at `arguments`, binds the associated type `projection` names (a
        projection carrying its member and any associated-type arguments) to
        `resolved`. `arguments` is the impl's substitution: one binding per
        parameter of the impl, keyed by the parameter as the impl spells it.
        It holds the arguments selection chose, so verification substitutes
        them rather than reading them off the projection: a parameter only a
        where-clause equality determines has no position in the projection to
        be read from. Every type parameter an argument spells is one the
        equality spells, so the instance a clone carries the equality to
        carries the arguments with it.

      * a `TraitApplicationAttr`, an obligation the impl supplies, discharging a
        cited conditional sibling impl's standing assumptions; verification
        resolves the named impl by symbol and never scans the module, so an
        impl's verdict does not depend on the unrelated impls the module carries.
        The impl is read at the application it names, so this arm carries no
        arguments.

    Printed `predicate by @impl[!P = T, ...]`; the brackets are omitted when
    the impl binds no parameter.

    Witnessing is one concept carried two ways, chosen by scope. This attribute
    IS the witness and stands alone as declaration-level witnessing. Where
    evidence must be a typed claim value, the `trait.witness` op wraps this same
    content and produces the value. See `trait.impl` for a printed `witnesses [...]`
    array carrying both arms.

    Both predicate arms are walk-visible, so every symbol either names is
    reached by the framework's own symbol-user driver wherever this attribute
    rides. A projection-resolution witness is immutable under rewriting for the
    reason `TypeEqualityAttr` states, and its arguments with it: every replacer
    the dialect builds carries the attribute-level rules that return an
    equality, and an equality-armed witness, unchanged and skip their interior.
    A clone moves the claim a `trait.witness` produces, never the witness it
    holds; verification carries the stored endpoints and arguments to the
    clone's equality by the one substitution relating the two.
  }];

  let parameters = (ins
    "::mlir::Attribute":$predicate,
    "::mlir::FlatSymbolRefAttr":$impl_ref,
    ArrayRefParameter<"TypeBindingAttr">:$arguments
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

// A predicate quantified over type parameters of its own, under premises of its
// own: a requirement a trait states for every argument of an associated type's
// parameters (a generic associated type's bound).
def BoundPredicateAttr : AttrDef<TraitDialect, "BoundPredicate",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "bound";
  let summary = "A predicate quantified over parameters of its own";
  let description = [{
    Represents `forall parameters. premises => conclusion`: for every argument
    of each parameter at which every premise holds, the conclusion holds. The
    premises and the conclusion are each a trait application or a type
    equality, spelled over the parameters the binder introduces and over the
    parameters of the declaration it stands in. It is the requirement a trait
    states for a generic associated type's bound: `type A<X>: Marker where X:
    Other` is

      forall [!X] where [@Other[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]

    The parameters are the binder's own: a reader instantiates them together
    with the declaration's parameters, in one substitution, and never reads one
    unbound. So each is a type parameter, they are distinct, and the conclusion
    spells at least one of them -- a binder quantifying nothing is a plain
    predicate spelled another way. Bound predicates do not nest.

    Printed `forall [params] where [premises] -> conclusion`; `where [...]` is
    omitted when there are no premises.
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$parameters,
    ArrayRefParameter<"::mlir::Attribute">:$premises,
    "::mlir::Attribute":$conclusion
  );

  let extraClassDeclaration = [{
    /// The substitution that instantiates this binder: each parameter it binds
    /// taking the argument at its position in `arguments`, which has one per
    /// parameter.
    SpecializationMap bindingFor(ArrayRef<Type> arguments) const;
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

// Evidence for one predicate under a bound requirement's binder, as an impl
// states it for the trait's bound requirement.
def BoundBodyAttr : AttrDef<TraitDialect, "BoundBody",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "body";
  let summary = "Evidence for a predicate under a bound requirement's binder";
  let description = [{
    Evidence that one predicate holds under the binder of a bound requirement,
    inside the impl that states it. Four forms, each citing by position:

      * `premise N` -- the binder's premise N;
      * `where N` -- entry N of the impl's own where clause;
      * `refl` -- an equality whose two sides are one type once read through
        the impl's own bindings;
      * `@impl[!P = T, ...] given [body, ...]` -- the impl named, at the
        arguments the bindings carry for its parameters, whose application is
        the predicate; `given` holds one body per entry of that impl's where
        clause, in its order, each discharging the entry at those arguments.
        It is omitted when that where clause is empty. The impl may be the
        one stating the evidence: a requirement it proves by being itself.

    Every predicate is compared after reading it through the stating impl's
    own associated-type bindings, so `Has[S]::A<X>` is the type the impl binds
    `A<X>` to.
  }];

  let parameters = (ins
    OptionalParameter<"std::optional<unsigned>">:$premise,
    OptionalParameter<"std::optional<unsigned>">:$where_entry,
    "bool":$refl,
    OptionalParameter<"::mlir::FlatSymbolRefAttr">:$impl_ref,
    OptionalArrayRefParameter<"TypeBindingAttr">:$arguments,
    OptionalArrayRefParameter<"BoundBodyAttr">:$discharges
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

// The evidence an impl states for one bound requirement of its trait.
def BoundEvidenceAttr : AttrDef<TraitDialect, "BoundEvidence",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "bound_evidence";
  let summary = "An impl's evidence for its trait's bound requirement";
  let description = [{
    The evidence an impl states for the bound requirement at position
    `requirement` of its trait's where clause: `predicate` is that requirement
    at the impl's own arguments -- the binder's parameters as the trait spells
    them, its premises and conclusion instantiated -- and `body` is evidence
    for the conclusion under the binder, which holds for every argument of the
    parameters at which the premises hold. The impl verifier refuses a
    predicate other than the requirement at that position, and a body that does
    not prove the conclusion.

    Printed `N: forall [...] where [...] -> conclusion by body`.
  }];

  let parameters = (ins
    "unsigned":$requirement,
    "BoundPredicateAttr":$predicate,
    "BoundBodyAttr":$body
  );

  let hasCustomAssemblyFormat = 1;
}

// An ordered array of the one-hop predicates a trait requires -- its `where`
// clause; each entry is a trait application, a type equality, or a bound
// predicate (see the description).
def PredicateArrayAttr : AttrDef<TraitDialect, "PredicateArray",
    [DeclareAttrInterfaceMethods<SymbolUserAttrInterface>]> {
  let mnemonic = "predicate_array";
  let summary = "An ordered array of predicates required by a trait";
  let description = [{
    Represents the ordered list of one-hop predicates a `trait.trait` requires
    (its `where` clause). Each entry is one of three arms, distinct in meaning:

      * a `TraitApplicationAttr` -- an implementation must exist for a trait
        applied to a list of types, e.g. `@Eq[!S]`;
      * a `TypeEqualityAttr` -- two spellings must name one type, e.g.
        `!trait.proj<@Trait[!S], "Assoc"> = i64`;
      * a `BoundPredicateAttr` -- a predicate that holds for every argument of
        parameters of its own, e.g. a generic associated type's bound
        `forall [!X] -> @Marker[!trait.proj<@Trait[!S], "A", [!X]>]`. Only a
        trait's where clause states one; an impl's premises bind nothing.

    Application entries are proved at impl selection; equality entries are
    checked at impl verification against the impl's own bindings; a bound
    entry is proved once, for every argument, by the evidence body the impl
    states for it. An application-only array prints and parses as a plain
    array of trait applications; existing textual forms round-trip unchanged.

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

    // The equality-arm entries only, in declaration order.
    llvm::SmallVector<TypeEqualityAttr> getEqualities() const {
      llvm::SmallVector<TypeEqualityAttr> equalities;
      for (::mlir::Attribute p : getPredicates())
        if (auto equality = mlir::dyn_cast<TypeEqualityAttr>(p))
          equalities.push_back(equality);
      return equalities;
    }

    // Whether any entry is an application requirement.
    bool hasApplications() const {
      return llvm::any_of(getPredicates(), [](::mlir::Attribute p) {
        return mlir::isa<TraitApplicationAttr>(p);
      });
    }

    // Whether any entry is an equality requirement.
    bool hasEqualities() const {
      return llvm::any_of(getPredicates(), [](::mlir::Attribute p) {
        return mlir::isa<TypeEqualityAttr>(p);
      });
    }

    // Whether any entry is a bound predicate.
    bool hasBoundPredicates() const {
      return llvm::any_of(getPredicates(), [](::mlir::Attribute p) {
        return mlir::isa<BoundPredicateAttr>(p);
      });
    }

    // Convenience: build from application entries alone (the common case).
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
