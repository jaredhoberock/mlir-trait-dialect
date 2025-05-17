#pragma once

#include <map>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::trait {

bool isPolymorph(func::FuncOp fn);

std::string mangleMethodName(StringRef traitName, Type receiverType, StringRef methodName);

std::string manglePolymorphicFunctionName(func::FuncOp polymorph,
                                          const DenseMap<Type, Type> &substitution);

func::FuncOp monomorphizeFunction(func::FuncOp polymorph,
                                  const DenseMap<Type, Type> &substitution);

func::FuncOp cloneAndSubstituteReceiverType(func::FuncOp method,
                                            Type receiverType);


LogicalResult applySubstitution(func::FuncOp polymorph,
                                const DenseMap<Type, Type> &substitution);

}
