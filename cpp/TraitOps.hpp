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

/// The projection-resolution certificate audit at the symbol seam, factored so
/// that both `WitnessOp::verifySymbolUses` and the C-API seam-audit query run
/// exactly this check on the same inputs. Looks the cited impl up in `module`,
/// resolves the `redex` projection through the impl's associated-type binding
/// specialized for the redex's trait application, applies the equality
/// `premises`, and compares the result against `contractum` receipt-blind.
/// Succeeds when the cited impl binds the redex to the contractum. `err` (never
/// null) receives the diagnostic on refusal; a caller that treats refusal as a
/// classification answer rather than an error suppresses it at the diagnostic
/// engine.
LogicalResult auditProjResolveCertificate(
    ModuleOp module, Type redex, Type contractum, FlatSymbolRefAttr citedImpl,
    ArrayRef<TypeEqualityAttr> premises,
    llvm::function_ref<InFlightDiagnostic()> err);

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


#define GET_OP_CLASSES
#include <TraitOps.hpp.inc>
