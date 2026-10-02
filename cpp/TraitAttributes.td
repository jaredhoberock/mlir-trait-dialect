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

#endif // TRAIT_ATTRIBUTES_TD
