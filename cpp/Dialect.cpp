#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"

#include "Dialect.cpp.inc"

namespace mlir::trait {

void TraitDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  registerTypes();
}

}
