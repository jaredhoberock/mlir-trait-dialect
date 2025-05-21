#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "Dialect.hpp.inc"

namespace mlir::trait {

struct ConvertToTraitInterface : DialectInterface {
  using DialectInterface::DialectInterface;

  virtual void populateConvertToTraitConversionPatterns(RewritePatternSet& patterns) = 0;

  static StringRef getInterfaceName() { return "ConvertToTraitInterface"; }

  static mlir::TypeID getInterfaceID() {
    return mlir::TypeID::get<ConvertToTraitInterface>();
  }
};

} // end mlir::trait
