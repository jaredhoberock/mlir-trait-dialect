#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include "Types.hpp"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include "Dialect.cpp.inc"

namespace mlir::trait {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateTraitToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void TraitDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  registerTypes();

  addInterfaces<
    ConvertToLLVMInterface
  >();
}

}
