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

    Endpoints relate types, so they admit no claim, proven or not, as a trait
    application admits no claim argument: a claim is a value's evidence, never
    a type. An equality relating two claims would let a coerce turn evidence of
    one application into evidence of another with no equality between their
    arguments, and one relating a claim to a value would turn evidence into a
    value; the constructor forbids both states.

    The endpoints are ordered; orientation is recorded for provenance
    only. Replay is direction-blind, so no symmetry construct exists.

    The endpoints are ordinary sub-elements: every walk reaches them, so a
    symbol sealed in one is reached by the framework's own symbol-user driver
    and by any walk that classifies content. What no rewrite may do is move
    one. Global type rewrites — proof stamping foremost — rewrite claim types
    wherever they can see them; an endpoint that received a stamped proof would
    be exactly the state the constructor forbids, so two individually-correct
    passes would kill a legal program. The attribute's sub-element replacement
    therefore answers it unchanged: endpoints are a leaf to every rewrite,
    upstream's replacers included, and reachable to a read. The one mover is
    the clone rule — one substitution applied to both endpoints atomically,
    rebuilt through this constructor, re-checked (`respellClaimPredicate`).
  }];

  let parameters = (ins "::mlir::Type":$lhs, "::mlir::Type":$rhs);

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Replacement of the endpoints as sub-elements answers this attribute
    /// unchanged, so no `AttrTypeReplacer` can move one; a writer constructs a
    /// new attribute. Shadows the storage base's hook, which
    /// `AbstractAttribute::get` reads off the concrete class.
    static auto getReplaceImmediateSubElementsFn() {
      return [](auto instance, ::llvm::ArrayRef<::mlir::Attribute>,
                ::llvm::ArrayRef<::mlir::Type>) -> ::mlir::Attribute {
        return instance;
      };
    }
  }];

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

    A citation's claims move by their variables alone
    (`respellClaimPredicate`), and its arguments are spelled as those claims
    are, so the arguments move by the same rule and by no other: the
    attribute's sub-element replacement answers it unchanged, so no replacer
    rewrites its interior, and the one mover is the clone rule, which builds a
    new attribute stamping each argument with the clone's variable bindings
    alone (`makeTypeReplacerFromSubstitution`). A projection an argument spells is
    never resolved in place; a template clone still stamps `@impl[!T]` to
    `@impl[i64]`. The arguments are ordinary sub-elements to a read, so a
    symbol one names is reached by the symbol-user driver.

    Its custom assembly form is the bracketed list after the impl's symbol,
    `@impl[i32, i64]`, elided where empty; standing alone it prints as
    `#trait.impl_args<[i32, i64]>`.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$types);

  let extraClassDeclaration = [{
    /// Replacement of the arguments as sub-elements answers this attribute
    /// unchanged, so no `AttrTypeReplacer` can write them; a writer constructs
    /// a new attribute. Shadows the storage base's hook, which
    /// `AbstractAttribute::get` reads off the concrete class.
    static auto getReplaceImmediateSubElementsFn() {
      return [](auto instance, ::llvm::ArrayRef<::mlir::Attribute>,
                ::llvm::ArrayRef<::mlir::Type>) -> ::mlir::Attribute {
        return instance;
      };
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif // TRAIT_ATTRIBUTES_TD
