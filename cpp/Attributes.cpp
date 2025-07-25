#include "Dialect.hpp"
#include "Attributes.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_ATTRDEF_CLASSES
#include "Attributes.cpp.inc"

namespace mlir::trait {

void TraitDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Attributes.cpp.inc"
  >();
}

} // end mlir::trait
