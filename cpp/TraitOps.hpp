#pragma once

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include "TraitTypes.hpp"

namespace mlir::OpTrait {

template<class ChildOp>
struct HasOnlyChildOp {
  template<class ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
  public:
    static LogicalResult verifyTrait(Operation* op) {
      for (auto &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &child : block) {
            if (!isa<ChildOp>(child))
              return op->emitOpError()
                     << "only " << ChildOp::getOperationName()
                     << " is allowed inside this op";
          }
        }
      }
      return success();
    }
  };
};

} // end mlir::OpTrait

#include "TraitOpInterfaces.hpp.inc"

#define GET_OP_CLASSES
#include "TraitOps.hpp.inc"
