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
    Represents a trait symbol applied to an ordered list of type arguments: the
    predicate of an application claim (`!trait.claim<@Trait[Args]>`) and the
    head a `trait.derive` derives.

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

    /// The trait this application names, read in the symbol table `from`
    /// stands in (or is) through `symbolTable`.
    FailureOr<TraitOp> getTrait(SymbolTableCollection &symbolTable, Operation *from,
                                llvm::function_ref<InFlightDiagnostic()> errFn = nullptr) const;

    TraitOp getTraitOrAbort(SymbolTableCollection &symbolTable, Operation *from,
                            const char* message) const;

    /// The trait this application names, once the application is verified
    /// to apply it (`verifySymbolUses`), refusing at `op` otherwise.
    FailureOr<TraitOp> getVerifiedTrait(Operation *op,
                                        SymbolTableCollection &symbolTable) const;

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

    Endpoints relate types, so an equality standing in a program admits no
    claim in either, proven or not, as a trait application admits no claim
    argument: a claim is a value's evidence, never a type. An equality relating
    two claims would let a coerce turn evidence of one application into
    evidence of another with no equality between their arguments, and one
    relating a claim to a value would turn evidence into a value. The rule is
    `verifyEndpointsRelateTypes`: every evidence op (allege, derive, project,
    witness, coerce) asks it of the evidence it reads and produces, at every
    verification (`EqualitiesRelateTypes`), so no coerce cites such an
    equality; the claim's symbol-use verifier asks it wherever an equality
    naming a symbol stands (`ClaimType::verifySymbolUses`; the symbol-use walk
    reaches no type naming none); and erasure's entry and selection refuse an
    equality whose sides, as written or as resolved, are either
    (`ImplResolver::readEqualitySides`). The constructor does not, because a
    clone's substitution and selection's normal forms rebuild an equality and
    neither can fail.

    The endpoints are ordered; orientation is recorded for provenance
    only. Replay is direction-blind, so no symmetry construct exists.

    The endpoints are ordinary sub-elements, which every walk reaches and a
    uniform rewrite (a rename, a clone's variable bindings) moves; an endpoint
    holds no claim, at every verification (`verifyEndpointsRelateTypes`). A
    rewrite normalizing at one operation leaves a claim's predicate as written,
    since its proof proves that spelling (`verifyCitation`), and the writers
    that normalize so install that rule through `addPredicateRules`.
  }];

  let parameters = (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs);

  let genVerifyDecl = 1;

  // Prints `lhs = rhs`. The leading empty literal drops the space the
  // declarative printer emits before a leading parameter, so an equality
  // printed directly into an enclosing `<...>` reads with no gap after the `<`.
  let assemblyFormat = "`` $lhs `=` $rhs";
}

// The arguments a citation states for its impl's parameters, by position: a
// derive's, and a projection-resolution witness's.
def ImplArgumentsAttr : AttrDef<TraitDialect, "ImplArguments", []> {
  let mnemonic = "impl_args";
  let summary = "The arguments a citation states for its impl's parameters";
  let description = [{
    The type each of a cited impl's parameters takes at one citation, in the
    order of the impl's parameters: argument `i` is what `!trait.poly<i>`
    stands for there. The citation's verifier instantiates the impl's header at
    them and compares the result with the claim the citation derives or
    resolves by identity, so the two must move together.

    The arguments are ordinary sub-elements, which every walk reaches and a
    uniform rewrite (a rename, a clone's variable bindings) moves, as it moves
    the claim they are spelled with. A rewrite normalizing at one operation
    leaves them as written, since the citation's verifier compares them with
    that claim by identity (`verifyCitationOf`), and the writers that normalize
    so install that rule through `addPredicateRules`.

    Its custom assembly form is the bracketed list after the impl's symbol,
    `@impl[i32, i64]`, elided where empty; standing alone it prints as
    `#trait.impl_args<[i32, i64]>`.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$types);

  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_ATTRIBUTES_TD
