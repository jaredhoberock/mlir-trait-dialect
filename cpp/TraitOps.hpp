// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include "TraitTypes.hpp"

namespace mlir::trait {

/// Refuses `op` unless every region from its own up to its nearest isolated
/// ancestor enforces SSA dominance, where it reads any value: a value it reads
/// is defined in one of those regions, so each is then defined before `op`
/// and none is computed from `op`'s own result.
LogicalResult verifyReadsUnderDominance(Operation *op);

/// Refuses `op` where one of its operands or results is evidence of an
/// equality relating evidence (`verifyEndpointsRelateTypes`).
LogicalResult verifyEqualitiesRelateTypes(Operation *op);

/// The evidence an evidence op reads and produces states no equality relating
/// evidence: an op verifier, which every verification runs, unlike the
/// symbol-use walk, which skips a type naming no symbol.
template <typename ConcreteType>
class EqualitiesRelateTypes
    : public ::mlir::OpTrait::TraitBase<ConcreteType, EqualitiesRelateTypes> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyEqualitiesRelateTypes(op);
  }
};

/// An op reading evidence: every value it reads is defined where SSA dominance
/// holds (`verifyReadsUnderDominance`), so a derivation through its operands is
/// a finite term and a walk reading evidence through them ends. A graph region
/// -- a module body -- would let the op read its own result, directly or
/// through other ops, even from inside a region nested in it.
template <typename ConcreteType>
class ReadsUnderDominance
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ReadsUnderDominance> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyReadsUnderDominance(op);
  }
};

/// Rewrite a type with every proven application claim stripped to its unproven
/// form: the claim the type is, or each claim it holds as a value's type, never
/// a claim's interior, which holds no claim. Coerce comparison is modulo the
/// proof, permanently.
Type stripClaimProofs(Type type);

/// Whether `lhs` and `rhs` are one type once each is stripped of its proofs
/// (`stripClaimProofs`). Two claims are compared by their predicates and two
/// function types position by position, so neither is rebuilt to be compared.
bool equalModuloProofs(Type lhs, Type rhs);

/// Whether `view`, the type a producer gives a position, settles `spelled`,
/// the type the position is spelled with: the two state one type modulo the
/// proofs their claims carry, and `view` names a proof at every application it
/// spells. A position settled so repeats its producer's evidence, and nothing
/// selects for it.
bool settles(Type view, Type spelled);

/// The evidence `value` carries: its proven application claim, or, where it is
/// a coercion's result, the evidence the coerced value carries; null where it
/// carries none yet. A coercion respells evidence and makes none, so a reader
/// of evidence reads it through every coercion, and no proof is written at a
/// spelling only such readers meet. The walk ends: a coerce reads its input
/// where SSA dominance holds (`ReadsUnderDominance`).
ClaimType evidenceOf(Value value);

/// The type a reader of evidence takes `value` at: the evidence it carries,
/// else its own type.
Type evidenceTypeOf(Value value);

/// The identity of the constructor a type is built by: every part of the type
/// that is not a child -- a function type's arity split, a vector's or memref's
/// shape, a memref's layout and memory space, a trait or associated-type name.
/// Two types denote the same constructor exactly when their keys are equal, so
/// distinct constructors never share a key and congruence over the children is
/// sound. A claim, a projection and a function type are keyed by the attributes
/// and counts they hold, so naming their constructor mints nothing; every other
/// type by its shell (`decomposeTerm`).
struct TermKey {
  enum class Kind : uint8_t { Type, Function, Application, Equality, Projection };
  Kind kind = Kind::Type;
  /// A `Type` kind's shell, or the type itself where it has none.
  Type type;
  /// The trait an application claim or a projection names, and the associated
  /// type a projection names.
  Attribute trait, associatedType;
  /// A function type's input count, a projection's associated-type argument
  /// count.
  unsigned split = 0;

  /// Whether the key names a type constructor, which may normalize its
  /// arguments when it builds a type: anything but a claim or a projection.
  bool isTypeConstructor() const {
    return kind == Kind::Type || kind == Kind::Function;
  }
  bool operator==(const TermKey &other) const {
    return kind == other.kind && type == other.type && trait == other.trait &&
           associatedType == other.associatedType && split == other.split;
  }
  bool operator!=(const TermKey &other) const { return !(*this == other); }
};

/// A type's term decomposition for ground reasoning: its constructor's identity
/// together with the positional type children the constructor is applied to.
struct TermShape {
  TermKey key;
  SmallVector<Type, 4> children;
};

/// Decompose a type into its constructor key and positional type children.
/// Claims, projections, and equality endpoints carry their type arguments inside
/// hand-written attribute storage the generic sub-element walk cannot see, so
/// each is enumerated explicitly, as a function type's inputs and results are.
/// Every other type derives its key by rebuilding itself with its immediate
/// sub-element types replaced by numbered placeholders: two containers share a
/// shell exactly when they differ only in their children. A constructor that
/// declines the placeholder arguments yields no shell; that type is keyed
/// atomically instead (see the guard in the definition).
TermShape decomposeTerm(Type t);

/// Whether `lhs` and `rhs` are equal under the ground congruence closure of the
/// premise equalities -- the one equational entailment decision the verifiers
/// make. The two types are compared as given, so a caller comparing modulo
/// proofs strips them first. Defined beside the closure.
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
  StringRef getName() const final { return "<Obligation>"; }
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

/// Refuses `standing`, an op standing under `name`, the name of the instance
/// a call needs, naming the call (`getOrCutInstance`): the caller names each
/// standing op once, however often the stage asks for the instance.
using InstanceNameRefusal =
    llvm::function_ref<void(Operation *standing, StringRef name)>;

} // end mlir::trait

#define GET_OP_CLASSES
#include <TraitOps.hpp.inc>

namespace mlir::trait {

/// The symbol of `owner`'s member function `name`, `<owner>.<name>`: the one
/// rule naming each member a trait declares and each function an impl
/// implements one by.
std::string getMemberSymbol(StringRef owner, StringRef name);

/// The name within `owner` of the member function `symbol` names, the inverse
/// of `getMemberSymbol`: `symbol` past `owner` and a dot; none where `symbol`
/// is not named under `owner`.
std::optional<StringRef> getMemberName(StringRef owner, StringRef symbol);

/// Whether `op` produces a claim whose evidence is read by position, off its
/// operands or off the declarations they name: a projection, a derive, a
/// coercion, or a call computing evidence. The stage proves such a result by
/// that reading -- the impl's returned evidence inlined at the projection, the
/// proof whose body the derive is, the input's proof respelled, the callee's
/// body inlined at the call -- and never by selecting on the result's
/// spelling.
bool producesPositionalEvidence(Operation *op);

/// The type arguments `call` supplies for `formal`, the signature of its
/// callee `callee`, whose labels lie below `bound`, once that signature
/// instantiated at them is `actual`, the signature the call states: read off
/// `actual` against `formal` as declared, the declaration's parameters and the
/// caller's labels read in one pass and never meeting in one type, through
/// `selection` where it is given, and refused through `err` where the two
/// disagree or a parameter no position determines.
///
/// `selection` resolves a type's projections through the stage's impl
/// selection, which at pass time is all both signatures are read through. A
/// verifier passes none and compares the two signatures as written: the call
/// spells its declaration's signature at its arguments, and a value spelled
/// otherwise reaches it through a coercion.
FailureOr<SpecializationMap>
readCallTypeArguments(Operation *call, FunctionType formal, unsigned bound,
                      FunctionType actual, StringRef callee,
                      llvm::function_ref<Type(Type)> selection,
                      llvm::function_ref<InFlightDiagnostic()> err = nullptr);

/// What a call of a trait's member reaches once its receiver -- the call's
/// first operand -- carries evidence: the function implementing the member,
/// read off the pair the impl the receiver's proof names, through the root it
/// rests on, makes for it.
struct MemberCall {
  /// The function the impl pairs with the member: the impl's own, or the
  /// member itself where the impl pairs it with nothing, leaving it to its
  /// trait's default. Null where the call names no trait's member.
  func::FuncOp function;
  /// The claim the root proves, naming the root: the impl's header at the
  /// arguments the root's derive states.
  ClaimType root;
  /// The root's derive, given the evidence for the impl's where entries; null
  /// for an unconditional impl named as its own proof, or a default.
  DeriveOp derive;
  /// How many where parameters `function` takes after the receiver: the
  /// impl's where entries for the impl's own function, none for a default.
  unsigned whereCount = 0;
};

/// Reads `call` as a call of a trait's member: one whose callee leads with an
/// unproven application of a trait listing the callee among its members.
/// Answers a `MemberCall` with no function where it is no such call; failure
/// where it is one whose receiver carries no evidence yet or whose proof names
/// no impl.
FailureOr<MemberCall> readMemberCall(FuncCallOp call,
                                     SymbolTableCollection &symbolTable);

/// Clones `member.function` at `rewriter`'s insertion point as `instanceName`,
/// stamped under `variables` and `projections`, with its where parameters
/// dropped: each one its body reads takes a clone of the premise the root's
/// derive is given at its position, so the instance is keyed and called by its
/// receiver and its other operands alone. Answers null where the function has
/// no body.
func::FuncOp cutMemberInstance(RewriterBase &rewriter, MemberCall member,
                               StringRef instanceName,
                               const SpecializationMap &variables,
                               const ProjectionBindings &projections);

/// Replaces `call`, a member call computing evidence, by the body of
/// `member.function` stamped under `subst`: its receiver parameter takes
/// `receiver`, its where parameters clones of the premises the root's derive
/// is given, its inputs the call's other operands, and the call's result the
/// operand the body returns. The claim it leaves is read by position, never
/// selected again.
LogicalResult inlineMemberAt(PatternRewriter &rewriter, FuncCallOp call,
                             MemberCall member, Value receiver,
                             const CallSubstitution &subst);

/// Ends `impl`'s body, an impl of `trait`, with a return of an allegation of
/// each of the trait's requirements at the impl's self application: the
/// evidence an impl generator that knows none of it supplies, which impl
/// selection proves where a use reads it. The allegations stand before the
/// return, whose operands they become.
void allegeRequirements(ImplOp impl, TraitOp trait, OpBuilder &builder);

/// Verifies that every function `impl` pairs with a member of `trait`, its
/// trait, is that member's declaration carried to the impl: the function's
/// type with the impl's block arguments dropped from its front is the member's
/// with the trait's self claim dropped, instantiated at the impl's arguments,
/// read through the impl's own bindings and where equalities (rustc's
/// `compare_impl_method`). A member the impl pairs with nothing is left to the
/// trait's default and is not compared. No pass rewrites a template, so the
/// comparison is made once, on the module as written, where the stage begins
/// (`verifyAcyclicTraits`); the impl's own verifier checks the pairs' shape.
LogicalResult verifyImplMemberSignatures(ImplOp impl, TraitOp trait,
                                         SymbolTableCollection &symbolTable);

} // end mlir::trait
