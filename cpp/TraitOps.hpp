// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include "SymbolLookup.hpp"
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

class NormalizationContext;

/// Rewrite a type with every proven application claim stripped to its unproven
/// form. Coerce comparison is modulo the proof, permanently.
Type stripClaimProofs(Type type);

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

namespace mlir::trait {

/// The obligations the stage has yet to discharge, as the resource of an
/// effect: `trait.allege` and `trait.project` write it until the stage
/// decides the allegation or inlines the projection, so neither is dead
/// while it stands, as `cf.assert` is not. It is no memory, so no analysis of
/// memory reads it.
struct ObligationResource
    : public SideEffects::Resource::Base<ObligationResource> {
  StringRef getName() final { return "<Obligation>"; }
};

/// What the evidence a projection of a proven claim reads stands on
/// (`ProjectOp::readEvidence`): the impls whose requirement returns it is read
/// through, in order, each at the application it is read at, and how the
/// reading ends.
struct EvidenceReading {
  enum class End {
    /// At evidence that is no projection: a derive, a witness, an
    /// allegation, a coercion of one -- a base, however long the reading.
    Base,
    /// At a requirement of an impl at an application read already: evidence
    /// defined by itself, which has no base.
    Cycle,
    /// Still projecting past the instantiation depth limit.
    Overflow,
  };
  End end = End::Base;
  SmallVector<StringAttr> impls;
  SmallVector<ObligationFrame> chain;
};

} // end mlir::trait

#define GET_OP_CLASSES
#include <TraitOps.hpp.inc>

namespace mlir::trait {

/// Whether `op` produces a claim whose evidence is read by position, off its
/// operands or off the declarations they name: a projection, a derive, a
/// coercion, or a call computing evidence. The stage proves such a result by
/// that reading -- the impl's returned evidence inlined at the projection, the
/// proof whose body the derive is, the input's proof, the method's body
/// inlined at the call -- and never by selecting on the result's spelling, so
/// neither its demand walk nor its respelling sweep reads that result.
bool producesPositionalEvidence(Operation *op);

/// Ends `impl`'s body, an impl of `trait`, with a return of an allegation of
/// each of the trait's requirements at the impl's self application: the
/// evidence an impl generator that knows none of it supplies, which impl
/// selection proves where a use reads it. The allegations stand before the
/// return, whose operands they become.
void allegeRequirements(ImplOp impl, TraitOp trait, OpBuilder &builder);

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

  /// A hypothesis in scope: `a` and `b` are the same type. An impl's own
  /// where-clause equalities are exactly these while its own obligations are
  /// checked -- an impl whose clause says `F::Output = Acc` satisfies a
  /// trait-header requirement spelled `F::Output = Acc` by that hypothesis and
  /// by nothing else, and a projection no hypothesis and no binding reduces is
  /// equal to itself alone.
  ///
  /// A hypothesis relates its two types; it does not rewrite the one into the
  /// other. Hypotheses spelled in opposite orientations put their endpoints in
  /// one class, and normalizing rewrites every member of a class to the one
  /// member the class stands for.
  void assumeEqual(Type a, Type b) { equalities.assumeEqual(a, b); }

  /// Also reads through impl selection, the context the stage holds:
  /// `selection` resolves the projections selection resolves in a type. A
  /// verifier sets none: what it may reduce a projection through is the
  /// evidence in front of it. `selection` must outlive this context.
  void setSelection(llvm::function_ref<Type(Type)> selection) {
    this->selection = selection;
  }

  /// XXX TODO Also reads the impls `module` holds, under `scope`. A verifier
  /// that sets this decides by the impls standing around it rather than by the
  /// evidence in front of it, which is what makes one verifier's verdict depend
  /// on an unrelated impl. Deleted once the evidence an op carries covers every
  /// spelling it must reduce: for `trait.derive`, once a claim operand that is
  /// neither proven nor derived carries the impl serving it; for a call,
  /// once the claim the call commits to carries the impls serving the
  /// projections its own arguments spell; and for a proof and a witness, once
  /// the declarations they read carry the evidence `LookupScope` names.
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
  TypeEquivalence equalities;
  llvm::function_ref<Type(Type)> selection;
  ModuleOp moduleLookup;
  LookupScope moduleLookupScope = LookupScope::Ground;
};

} // end mlir::trait
