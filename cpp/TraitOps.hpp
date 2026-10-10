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

/// Rewrite a type with every proven application claim stripped to its unproven
/// form: the claim the type is, or each claim it holds as a value's type, never
/// a claim's interior, which holds no claim. Coerce comparison is modulo the
/// proof, permanently.
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
/// and trait.coerce's proven arm share. The two types are compared as given, so
/// a caller comparing modulo proofs strips them first. Defined beside the
/// closure; declared here for the composition arm's verifier.
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
/// proof whose body the derive is, the input's proof respelled, the method's
/// body inlined at the call -- and never by selecting on the result's
/// spelling, so the stage's sweep proves no claim that result spells.
bool producesPositionalEvidence(Operation *op);

/// Ends `impl`'s body, an impl of `trait`, with a return of an allegation of
/// each of the trait's requirements at the impl's self application: the
/// evidence an impl generator that knows none of it supplies, which impl
/// selection proves where a use reads it. The allegations stand before the
/// return, whose operands they become.
void allegeRequirements(ImplOp impl, TraitOp trait, OpBuilder &builder);

} // end mlir::trait
