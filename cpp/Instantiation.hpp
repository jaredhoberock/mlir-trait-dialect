#pragma once

#include "Ops.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>

namespace mlir::trait {

bool isPolymorph(func::FuncOp fn);

std::string mangleMethodName(StringRef traitName, Type receiverType, StringRef methodName);

std::string mangleFunctionName(StringRef name,
                               const DenseMap<Type, Type> &substitution);

func::FuncOp instantiatePolymorph(OpBuilder& builder,
                                  func::FuncOp polymorph,
                                  StringRef instanceName,
                                  const DenseMap<Type,Type> &substitution);

ImplOp instantiatePolymorphicImpl(OpBuilder& builder,
                                  ImplOp polymorph,
                                  Type receiverType);

void instantiatePolymorphicRegion(OpBuilder& builder,
                                  Region& polymorph,
                                  Region& monomorph,
                                  const DenseMap<Type,Type> &substitution);

}
