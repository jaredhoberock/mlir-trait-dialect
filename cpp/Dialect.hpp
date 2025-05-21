#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "Dialect.hpp.inc"

namespace mlir::trait {

struct ConvertToTraitPatternInterface : DialectInterface {
  inline ConvertToTraitPatternInterface(Dialect *dialect)
    : DialectInterface(dialect, TypeID::get<ConvertToTraitPatternInterface>())
  {}

  virtual void populateConvertToTraitConversionPatterns(RewritePatternSet& patterns) const = 0;

  inline static StringRef getInterfaceName() { return "ConvertToTraitInterface"; }

  inline static ::mlir::TypeID getInterfaceID() {
    return ::mlir::TypeID::get<ConvertToTraitPatternInterface>();
  }
};

} // end mlir::trait
