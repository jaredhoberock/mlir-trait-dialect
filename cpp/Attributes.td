#ifndef TRAIT_ATTRIBUTES_TD
#define TRAIT_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

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
    "::mlir::FlatSymbolRefAttr":$trait,  // referenced trait symbol
    ArrayRefParameter<"Type">:$type_args // ordered list of type arguments
  );

  let assemblyFormat = "$trait `[` $type_args `]`";
}

#endif // TRAIT_ATTRIBUTES_TD
