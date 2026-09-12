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

/// Verifies an equality-armed projection-resolution `witness` against `module`;
/// passing an application-armed witness is a caller bug. Succeeds iff the cited
/// impl (`witness.getImplRef()`), specialized for the projection's application
/// and modulo the equality `premises`, binds the projection to the resolved
/// type, proofs ignored. The cited impl's own assumptions must each be covered
/// by the application-arm `obligationPremises` -- deliberately not its trait
/// requirements, which may quantify over GAT variables with no ground instance
/// here. This use-site entry resolves the actual side's ground projections by
/// module lookup. `err`, when non-null, receives the diagnostic on refusal.
LogicalResult verifyProjectionResolutionAtUse(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr,
    TypeEqualityAttr currentEquality = {});

/// The ImplOp-verification companion to `verifyProjectionResolutionAtUse`, running the
/// same binding check and assumption discharge, differing in three ways. Its
/// head match is rigid -- only the cited impl's own generics instantiate -- so
/// the verdict is estate-independent. Its assumptions may also be covered by a
/// `dischargeWitnesses` entry, recursively over the same finite list. And on
/// success it returns the head-match substitution.
FailureOr<SpecializationMap> verifyProjectionResolutionAtImpl(
    ModuleOp module, WitnessAttr witness,
    ArrayRef<TypeEqualityAttr> premises,
    ArrayRef<TraitApplicationAttr> obligationPremises,
    ArrayRef<WitnessAttr> dischargeWitnesses,
    llvm::function_ref<InFlightDiagnostic()> err = nullptr);

/// Rewrite a type with every proven application claim stripped to its unproven
/// form. Coerce comparison is modulo the proof, permanently.
Type stripClaimProofs(Type type);

/// The pending judgment a marked (unproven) coerce carries; one judgment serves
/// every checker of this evidence. The endpoints must unify, giving every
/// `!trait.proj` term a shared variable keyed by the projection itself: the same
/// projection is one variable, every other position is rigid, and a whole
/// projection is opaque (its arguments are not descended). A projection may
/// resolve to any type the unification reaches -- a projection-free position,
/// itself, another bare projection, or a composite that still carries
/// projections -- since every projection standing in a binding is itself a
/// variable still owed a grounding at discharge, so a terminal that still
/// carries one is the weaker assertion. The one binding refused is one that
/// closes a cycle (an unfoundable infinite type), caught by an occurs check.
/// Ground truth is arbitrated later: the bonded erase pass judges the op once
/// monomorphization grounds every projection, refusing a coerce whose ground
/// endpoints stand apart. Endpoints arrive with proofs already stripped. `err`,
/// when non-null, receives the diagnostic on refusal.
LogicalResult verifyPendingProjectionUnification(
    Type input, Type result,
    llvm::function_ref<InFlightDiagnostic()> emitError = nullptr);

/// A type's term decomposition for ground reasoning: an exact constructor
/// identity together with the positional type children the constructor is
/// applied to. Two types denote the same constructor exactly when their keys are
/// equal; the key carries every part of a type that is not a child -- a function
/// type's arity split, a vector's or memref's shape, a memref's layout and memory
/// space, a trait or associated-type name -- so distinct constructors never share
/// a key and congruence over the children is sound.
struct TermShape {
  Attribute key;
  SmallVector<Type, 4> children;
};

/// Decompose a type into its constructor key and positional type children.
/// Claims, projections, and equality endpoints carry their type arguments inside
/// hand-written attribute storage the generic sub-element walk cannot see, so
/// each is enumerated explicitly. Every other type derives its key by rebuilding
/// itself with its immediate sub-element types replaced by numbered placeholders:
/// the resulting shell holds the full non-child storage by construction and
/// compares by exact type equality, so two containers share a key exactly when
/// they differ only in their children. A constructor that declines the
/// placeholder arguments yields no shell; that type is keyed atomically instead
/// (see the guard in the definition).
TermShape decomposeTerm(Type t);

/// Whether `lhs` and `rhs` are equal under the ground congruence closure of the
/// premise equalities -- the one entailment decision the witness composition arm
/// and trait.coerce's proven arm share. Defined beside the closure; declared here
/// for the composition arm's verifier.
bool entailedByGroundCongruence(Type lhs, Type rhs,
                                ArrayRef<TypeEqualityAttr> premises);

/// Refuses every type parameter `function`'s body mentions that its declaration
/// does not bind, reporting at the function with a note at the first mention.
///
/// A declaration binds the generics of its own signature, and, for a method, the
/// generics of the trait or impl header it is written in. Its body may mention
/// no others: a substitution is built from the declaration's parameters, so a
/// parameter the declaration does not bind has no source for its argument and
/// survives into whatever the body is cloned into. A nested declaration isolated
/// from above -- a nested function, a trait, an impl, a proof -- is left for its
/// own check. A region an op runs at run time (an `scf.if`, a `cf` block, a
/// cooperative body) stands in this scope, and a callable region the scope
/// reaches into -- a lambda a dialect specializes per use, such as a `tuple.map`
/// body -- binds the generics of its own argument and result types on top of the
/// ones already in scope. The type parameters a generic call spells for its
/// callee are read as the callee's and not as a mention here.
///
/// This is a whole-function judgment, not an op verifier: a step that reads or
/// clones a body asks it at its entry.
LogicalResult verifyFunctionBodyIsWellScoped(func::FuncOp function);

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

namespace mlir::trait {

/// One local associated-type resolution rule available while normalizing a type.
///
/// The rule says that projections whose trait application is exactly `app` may
/// be resolved through `impl` after applying `subst` to the impl's associated
/// type binding. It represents evidence already present at the current IR
/// boundary; it does not perform global impl lookup.
struct LocalProjectionRule {
  ImplOp impl;
  TraitApplicationAttr app;
  SpecializationMap subst;
};

/// Context controlling how far `normalize` may resolve projection types.
///
/// The small core of normalization is deliberately local: callers add explicit
/// evidence-derived rules, and projection heads not justified by those rules are
/// preserved. Global resolver-backed normalization remains a separate lowering
/// concern.
class NormalizationContext {
public:
  void addLocalProjectionRule(ImplOp impl, TraitApplicationAttr app,
                              const SpecializationMap &subst) {
    localProjectionRules.push_back({impl, app, subst});
  }

  /// A hypothesis in scope: wherever `from` stands, `to` stands. An impl's own
  /// where-clause equalities are exactly these while its own obligations are
  /// checked -- an impl whose clause says `F::Output = Acc` satisfies a
  /// trait-header requirement spelled `F::Output = Acc` by that hypothesis and
  /// by nothing else, and a projection no hypothesis and no binding reduces is
  /// equal to itself alone.
  void addEqualityRule(Type from, Type to) { equalityRules[from] = to; }

  /// Also reads what impl selection has settled, which is the context the stage
  /// holds on top of the evidence an op carries. A verifier sets none: what it
  /// may reduce a projection through is the evidence in front of it.
  void setRecordedFacts(const ReadOnlyImplResolver *reading) {
    recordedFacts = reading;
  }

  /// XXX TODO Also reads the impls `module` holds, under `scope`. A verifier
  /// that sets this decides by the impls standing around it rather than by the
  /// evidence in front of it, which is what makes one verifier's verdict depend
  /// on an unrelated impl. Deleted once the evidence an op carries covers every
  /// spelling it must reduce: for `trait.derive`, once a claim operand that is
  /// neither proven nor derived carries the impl serving it; for a call,
  /// once the claim the call commits to carries the impls serving the
  /// projections its own arguments spell.
  void setModuleLookup(ModuleOp module, LookupScope scope) {
    moduleLookup = module;
    moduleLookupScope = scope;
  }

  /// Resolves projections in `ty` using this context's local rules.
  ///
  /// The walk runs to a fixed point so a resolved associated type can expose
  /// another projection resolvable by the same local evidence.
  FailureOr<Type> normalize(
      Type ty,
      llvm::function_ref<InFlightDiagnostic()> err);

  /// Resolves projections in each input and result type of `functionType`.
  FailureOr<FunctionType> normalize(
      FunctionType functionType,
      llvm::function_ref<InFlightDiagnostic()> err);

private:
  SmallVector<LocalProjectionRule, 4> localProjectionRules;
  llvm::DenseMap<Type, Type> equalityRules;
  const ReadOnlyImplResolver *recordedFacts = nullptr;
  ModuleOp moduleLookup;
  LookupScope moduleLookupScope = LookupScope::Ground;
};

} // end mlir::trait
