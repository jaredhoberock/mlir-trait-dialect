#pragma once

#include "TraitOps.hpp"
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>

namespace mlir::trait {

AttrTypeReplacer makeTypeReplacerFromSubstitution(const DenseMap<Type,Type> &subst);

func::FuncOp instantiatePolymorph(OpBuilder& builder,
                                  func::FuncOp polymorph,
                                  StringRef instanceName,
                                  const DenseMap<Type,Type> &substitution);

void instantiatePolymorphicRegion(OpBuilder& builder,
                                  Region& polymorph,
                                  Region& monomorph,
                                  const DenseMap<Type,Type> &substitution);

}
