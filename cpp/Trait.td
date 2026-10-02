// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRAIT_DIALECT_TD
#define TRAIT_DIALECT_TD

include "mlir/IR/OpBase.td"

def TraitDialect : Dialect {
  let name = "trait";
  let summary = "A dialect for trait-constrained polymorphism";
  let cppNamespace = "::mlir::trait";

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();

    /// Parse an attribute of this dialect, refusing text its body leaves
    /// unread.
    ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type type) const override;
    void printAttribute(::mlir::Attribute attr,
                        ::mlir::DialectAsmPrinter &printer) const override;

    /// The requirements of the trait whose self application is `self` and
    /// whose requirement list is `requirements`, instantiated at the
    /// application `actual`, where an earlier instantiation recorded them.
    /// The instance is a function of those three interned values alone, so
    /// one recorded for any trait declaring them answers for every other.
    std::optional<::llvm::SmallVector<::mlir::Type>>
    lookupRequirementInstance(::mlir::Attribute self,
                              ::mlir::Attribute requirements,
                              ::mlir::Attribute actual) const;

    /// Records `instance` as the requirements `lookupRequirementInstance`
    /// answers with for `self`, `requirements` and `actual`.
    void recordRequirementInstance(::mlir::Attribute self,
                                   ::mlir::Attribute requirements,
                                   ::mlir::Attribute actual,
                                   ::llvm::ArrayRef<::mlir::Type> instance) const;

  private:
    /// The recorded requirement instances. Verifiers on worker threads read
    /// and record them, so every access holds the lock.
    mutable ::llvm::sys::SmartMutex<true> requirementInstanceLock;
    mutable ::llvm::DenseMap<
        std::tuple<::mlir::Attribute, ::mlir::Attribute, ::mlir::Attribute>,
        ::llvm::SmallVector<::mlir::Type, 2>>
        requirementInstances;

  public:
  }];
}

class TraitDialectOp<string mnemonic, list<Trait> traits = []> :
  Op<TraitDialect, mnemonic, traits>;

#endif // TRAIT_DIALECT_TD
