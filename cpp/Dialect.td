#ifndef TRAIT_DIALECT_TD
#define TRAIT_DIALECT_TD

include "mlir/IR/OpBase.td"

def TraitDialect : Dialect {
  let name = "trait";
  let summary = "A dialect for trait-constrained polymorphism";
  let cppNamespace = "::mlir::trait";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class TraitDialectOp<string mnemonic, list<Trait> traits = []> :
  Op<TraitDialect, mnemonic, traits>;

#endif // TRAIT_DIALECT_TD
