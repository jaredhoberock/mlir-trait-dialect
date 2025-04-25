#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace trait {

void populateTraitToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                           RewritePatternSet& patterns);
}
}
