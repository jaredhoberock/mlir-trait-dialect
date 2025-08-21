#ifndef TRAIT_ATTRIBUTES_TD
#define TRAIT_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "Trait.td"

def TraitApplicationAttr : AttrDef<TraitDialect, "TraitApplication"> {
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
    static TraitApplicationAttr get(MLIRContext *ctx, FlatSymbolRefAttr trait_name, ArrayAttr typesAttr) {
      SmallVector<Type, 4> types;
      types.reserve(typesAttr.size());
      for (TypeAttr ta : typesAttr.getAsRange<TypeAttr>()) {
        types.push_back(ta.getValue());
      }
      return TraitApplicationAttr::get(ctx, trait_name, types);
    }

    FailureOr<TraitOp> getTrait(ModuleOp module, llvm::function_ref<InFlightDiagnostic()> errFn = nullptr) const;

    TraitOp getTraitOrAbort(ModuleOp module, const char* message) const;

    bool mentionsType(Type ty) const {
      return llvm::any_of(getTypeArgs(), [&](Type arg) { return arg == ty; });
    }

    LogicalResult verifyTraitApplication(
        ModuleOp module,
        llvm::function_ref<InFlightDiagnostic()> emitError
    ) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def ConstraintsAttr : AttrDef<TraitDialect, "Constraints"> {
  let mnemonic = "requirements";
  let summary = "A list of prerequisite trait applications required by traits or assumed by implementation";
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
    LogicalResult verifyTraitApplications(
        ModuleOp module,
        llvm::function_ref<InFlightDiagnostic()> emitError) const {
      for (const auto& app : getApplications()) {
        if (failed(app.verifyTraitApplication(module, emitError)))
          return failure();
      }
      return success();
    }

    auto getApplicationsOf(FlatSymbolRefAttr traitName) const {
      return llvm::make_filter_range(
        getApplications(),
        [traitName](TraitApplicationAttr app) {
          return app.getTraitName() == traitName;
        }
      );
    }

    // make ConstraintsAttr behave like a range of TraitApplicationAttr
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
