#pragma once

#include <map>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::trait {

bool isPolymorph(func::FuncOp fn);

std::string manglePolymorphicFunctionName(func::FuncOp polymorph,
                                          const std::map<unsigned int, Type> &substitution);

// XXX it might be better if the substitution was just a mapping Type -> Type
func::FuncOp monomorphizeFunction(func::FuncOp polymorph,
                                  const std::map<unsigned int, Type> &substitution);

func::FuncOp cloneAndMonomorphizeSelfType(func::FuncOp method,
                                          Type concreteSelfType);


LogicalResult applySubstitution(func::FuncOp polymorph,
                                const std::map<unsigned int, Type> &substitution);

}
