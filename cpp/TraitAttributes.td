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

#endif // TRAIT_ATTRIBUTES_TD
