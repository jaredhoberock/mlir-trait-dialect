#pragma once

#include "TraitOps.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>

namespace mlir::trait {

bool isPolymorph(func::FuncOp fn);

func::FuncOp instantiatePolymorph(OpBuilder& builder,
                                  func::FuncOp polymorph,
                                  StringRef instanceName,
                                  const DenseMap<Type,Type> &substitution);

ImplOp instantiatePolymorphicImpl(OpBuilder& builder,
                                  ImplOp polymorph,
                                  ArrayRef<Type> typeArgs);

void instantiatePolymorphicRegion(OpBuilder& builder,
                                  Region& polymorph,
                                  Region& monomorph,
                                  const DenseMap<Type,Type> &substitution);

}
