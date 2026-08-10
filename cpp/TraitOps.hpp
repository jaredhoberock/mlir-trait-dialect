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
///
/// When `rigidHeadMatch` is set, the head match instantiates ONLY the cited
/// impl's own generics; the redex's application (the actual side) stays rigid, so
/// no module-visible impl resolves a projection spelled there and the audit
/// verdict never depends on the unrelated impls the module carries. The impl
/// birth audit sets it, so an impl's verdict is estate-independent. Left unset
/// (the default), the head match resolves the actual side's ground projections
/// by module lookup, as a witness-site audit does. When `outSubst` is non-null it
/// receives the head-match substitution, so a caller replaying the premise reuses
/// this build rather than deriving it a second time.
///
/// When `dischargeObligations` is set, the audit additionally requires the cited
/// impl's own assumptions -- specialized for the redex's application -- each to
/// be discharged: either by a hypothetical cover from the application-arm
/// `obligationPremises` (the citing impl's own where clause), or by a
/// `dischargeCitations` entry whose spelled application is the assumption and
/// whose named impl, specialized for it, has each of its own assumptions
/// discharged in turn over the same finite citation list. Both compare
/// receipt-stripped and modulo the equality `premises`. It deliberately does not
/// reach the cited impl's trait requirements, which may quantify over GAT
/// variables with no ground instance at the witness; requirement discharge
/// belongs to the proof and birth machinery. The obligation mode is off by
/// default -- the verifier stays binding-only -- and is turned on only by the
/// obligation-mode seam-audit query.
LogicalResult auditProjResolveCertificate(
    ModuleOp module, Type redex, Type contractum, FlatSymbolRefAttr citedImpl,
    ArrayRef<TypeEqualityAttr> premises,
    llvm::function_ref<InFlightDiagnostic()> err,
    ArrayRef<TraitApplicationAttr> obligationPremises = {},
    bool dischargeObligations = false,
    ArrayRef<DischargeCitationAttr> dischargeCitations = {},
    bool rigidHeadMatch = false,
    SpecializationMap *outSubst = nullptr);

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
