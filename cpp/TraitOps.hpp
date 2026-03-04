// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include "TraitTypes.hpp"

namespace mlir::trait {

/// Mixin that adds a `getModule()` convenience method to any op.
template <typename ConcreteType>
class HasGetModule : public ::mlir::OpTrait::TraitBase<ConcreteType, HasGetModule> {
public:
  FailureOr<ModuleOp> getModule(
      llvm::function_ref<InFlightDiagnostic()> err = nullptr) {
    auto module = this->getOperation()->template getParentOfType<ModuleOp>();
    if (!module) {
      if (err) err() << "not in a module";
      return failure();
    }
    return module;
  }
};

} // end mlir::trait

namespace mlir::OpTrait {

template<class... ChildOps>
struct HasOnlyChildOps {
  template<class ConcreteOp>
  class Impl : public mlir::OpTrait::TraitBase<ConcreteOp, Impl> {
  public:
    static LogicalResult verifyTrait(Operation* op) {
      for (auto &region : op->getRegions())
        for (auto &block : region)
          for (auto &child : block)
            if (!isa<ChildOps...>(child))
              return op->emitOpError() << "unexpected child op '"
                     << child.getName() << "'";
      return success();
    }
  };
};

} // end mlir::OpTrait

#include "TraitOpInterfaces.hpp.inc"

#define GET_OP_CLASSES
#include "TraitOps.hpp.inc"
