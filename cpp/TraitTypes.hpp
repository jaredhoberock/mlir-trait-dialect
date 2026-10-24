// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "TraitAttributes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>

namespace mlir { class OpBuilder; }

namespace mlir::trait {

// Generated interface declarations below mention these types before the
// concrete helper classes are defined in this header.
class TraitOp;
class SpecializationMap;
class CallSubstitution;
class ImplResolver;

}

#include <TraitTypeInterfaces.hpp.inc>

#define GET_TYPEDEF_CLASSES
#include <TraitTypes.hpp.inc>

namespace mlir { class AsmParser; }

namespace mlir::trait {

/// Parse one where-clause predicate: an application `@Trait[types...]` yielding a
/// TraitApplicationAttr, or an equality `!A = !B` yielding a checked
/// TypeEqualityAttr, disambiguated by the leading `@`. This is the single grammar for a where-clause predicate; a
/// `by @proof` tail (allowed only on an application claim) is the caller's to
/// add. Fails on a malformed predicate, having emitted the diagnostic where the
/// endpoints are ill-formed.
FailureOr<Attribute> parseApplicationOrEqualityPredicate(AsmParser &p);

/// The clone rule for a claim's predicate: rebuild `claim` with `respell`
/// applied to each endpoint of an equality, or to each type argument of an
/// application, atomically, through the checked constructor; an application
/// keeps the proof it names. Always skips the result's interior -- a predicate
/// moves through this rule or not at all, so a replacer registering it must
/// not let its own walk reach the predicate afterwards.
///
/// A claim states a proposition and names the proof of exactly that spelling
/// (`verifyCitation`), so a rewrite resolving a projection inside one would
/// leave the proof naming a spelling the claim no longer has. A clone moves a
/// predicate by its variables alone; the evidence for another spelling is a
/// proof of that spelling.
inline std::optional<std::pair<Type, WalkResult>> respellClaimPredicate(
    ClaimType claim, llvm::function_ref<Type(Type)> respell) {
  MLIRContext *ctx = claim.getContext();
  if (auto eq = claim.getEqualityAttr()) {
    Type newLhs = respell(eq.getLhs());
    Type newRhs = respell(eq.getRhs());
    if (newLhs == eq.getLhs() && newRhs == eq.getRhs())
      return std::make_pair(Type(claim), WalkResult::skip());
    return std::make_pair(Type(ClaimType::getEquality(ctx, newLhs, newRhs)),
                          WalkResult::skip());
  }
  TraitApplicationAttr app = claim.getTraitApplication();
  SmallVector<Type> args;
  for (Type arg : app.getTypeArgs())
    args.push_back(respell(arg));
  if (llvm::equal(args, app.getTypeArgs()))
    return std::make_pair(Type(claim), WalkResult::skip());
  return std::make_pair(
      Type(ClaimType::get(ctx,
                          TraitApplicationAttr::get(ctx, app.getTraitName(), args),
                          claim.getProof())),
      WalkResult::skip());
}

/// Whether a reading through selection reads `proj`: it is ground. A
/// polymorphic projection stands for as many types as its variables have
/// instances, so no resolver owes it an answer.
bool readsGroundProjection(ProjectionType proj);

/// Whether a head-keyed reading reads `proj`: its head application is ground.
/// Which impl serves a projection is settled by its head application, and what
/// that impl binds is a function of the projection's own associated-type
/// arguments, so such a projection is read whatever variables those arguments
/// still carry. A head still carrying a variable stands for as many impls as
/// that variable has instances.
bool readsGroundHeadProjection(ProjectionType proj);

/// Whether some part of `root`, `root` included, is polymorphic: a part
/// whose `PolymorphicTypeInterface` says so. One walk visits each distinct
/// part once.
bool isPolymorphicType(Type root);

/// How a substitution reaches a claim's predicate: through every binding it
/// holds, or through its variable bindings alone (`respellClaimPredicate`), as
/// a clone stamps one.
enum class ClaimPredicates { Substituted, VariablesAlone };

class SpecializationMap;

/// The value a substitution binds to a key that is no type variable -- a
/// projection, a proven claim -- or nothing where it binds none.
using OtherBindings = llvm::function_ref<std::optional<Type>(Type)>;

inline Type applySubstitution(const SpecializationMap &specialization,
                              OtherBindings others, Type root,
                              ClaimPredicates claims);
inline Type applySubstitutionToFixedPoint(
    const SpecializationMap &specialization, OtherBindings others, Type ty,
    ClaimPredicates claims);

/// The values a substitution binds to keys of one kind: each key bound to the
/// one value it stands for. A key is bound once, and binding it again to a
/// different value is a caller bug the assertion names.
template <typename KeyT, typename ValueT = Type>
class TypeBindings {
public:
  std::optional<ValueT> lookup(KeyT key) const {
    auto it = bindings.find(key);
    if (it == bindings.end())
      return std::nullopt;
    return it->second;
  }

  void bind(KeyT key, ValueT value) {
    assert((!bindings.count(key) || bindings.lookup(key) == value) &&
           "a binding must not be replaced with a different value");
    bindings[key] = value;
  }

  size_t bindingCount() const { return bindings.size(); }

protected:
  llvm::DenseMap<KeyT, ValueT> bindings;
};

/// The type arguments chosen for a declaration's type parameters, by label: slot
/// `i` holds the argument of the parameter `!trait.poly<i>`, which every
/// occurrence of that label stands for (`GenericTypeInterface::getParameterAtom`),
/// or nothing where none is chosen. A label is a position in the declaration
/// that binds it, so a lookup is an index.
class SpecializationMap {
public:
  std::optional<Type> lookup(GenericTypeInterface key) const {
    unsigned label = labelOf(key);
    if (label >= arguments.size() || !arguments[label])
      return std::nullopt;
    return arguments[label];
  }

  /// Binds `key`'s label to `value`. A label is bound once, and binding it
  /// again to a different value is a caller bug the assertion names.
  void bind(GenericTypeInterface key, Type value) {
    unsigned label = labelOf(key);
    if (label >= arguments.size())
      arguments.resize(label + 1);
    assert((!arguments[label] || arguments[label] == value) &&
           "a binding must not be replaced with a different value");
    arguments[label] = value;
  }

  size_t bindingCount() const {
    return llvm::count_if(arguments, [](Type argument) { return argument; });
  }

  /// The argument of each label below the bound, null where none is chosen.
  ArrayRef<Type> getArguments() const { return arguments; }

  // A specialization is fully composed by construction, so one structural
  // substitution pass is enough.
  //
  // Applying a substitution resolves nothing. Stamping a concrete argument into
  // a projection spelling can turn a symbolic projection into a ground one, and
  // the result carries that projection still spelled as written: what resolves
  // it is a later reading through the caller's established context, or a stamp
  // through the module-capable replacer.
  Type apply(Type ty) const {
    return applySubstitution(*this, nullptr, ty, ClaimPredicates::Substituted);
  }

  /// The specialization binding the declaration parameter labelled `i` to
  /// `arguments[i]`: a citation's argument list, stated by position.
  template <typename RangeT>
  static SpecializationMap fromPositions(RangeT &&arguments);

private:
  /// The label of the parameter `key` is an occurrence of.
  static unsigned labelOf(GenericTypeInterface key) {
    return cast<PolyType>(key.getParameterAtom()).getLabel();
  }

  SmallVector<Type, 4> arguments;
};

/// The concrete associated types projections stand for.
using ProjectionBindings = TypeBindings<ProjectionType>;

/// Whether a spelling is one nothing but a respelling can move.
///
/// Two things leave a spelling open. A ground projection is one the impls
/// standing when it was read could not resolve, so an impl generated since may
/// resolve it and reach a different type. A type variable is one the template it
/// belongs to binds differently for each instance, so what it spells is a
/// coincidence of the instance being read rather than a fact. A spelling with
/// neither has no open question in it, and a comparison of two such spellings is
/// a verdict nothing later revises.
inline bool spellingIsSettled(Type ty) {
  if (isPolymorphicType(ty))
    return false;
  bool open = false;
  ty.walk([&](Type sub) {
    if (isa<ProjectionType>(sub))
      open = true;
  });
  return !open;
}

/// Whether an equality premise read at a citation is one that citation cannot
/// decide, so the instances made of it decide it instead.
///
/// A reading carrying a type variable is a template's: the variable stands for
/// whatever each instance binds it to, and the instance is where the premise is
/// read. A reading with no variable in it is decided here even where it spells a
/// projection nothing resolves -- no instance moves that spelling either, so a
/// premise it leaves unequal is a premise that does not hold, and the impl
/// stating it does not apply. This is the judgment impl selection reads a
/// candidate's equality premises by.
inline bool premiseDefersToInstances(Type lhs, Type rhs) {
  return isPolymorphicType(lhs) || isPolymorphicType(rhs);
}

/// What impl selection resolves a ground projection to, failing where it
/// refuses the projection's application.
using ProjectionResolver = llvm::function_ref<FailureOr<Type>(ProjectionType)>;

/// CallSubstitution: SpecializationMap + ProjectionBindings.
///
/// The complete set of type rewrites needed to lower one call site, closed under
/// the projections those rewrites expose.
///
/// The factory below is the only way to make one, so a substitution that exists
/// is one `resolve` closed: every monomorphic projection the call spells is
/// bound to what impl selection resolves it to. A claim's proof is no binding: each
/// parameter of the instance a call lowers to takes its evidence from the
/// position it was supplied at, and a value its body computes from the value
/// that supplies it.
class CallSubstitution {
public:
  /// The closed substitution that lowers a call whose operands and results are
  /// `operandTypes` and `resultTypes` and whose callee signature is `formalTy`,
  /// starting from the arguments the call supplies for the callee's parameters.
  ///
  /// A projection binding can expose another projection, so discovery runs
  /// until no binding is added.
  ///
  /// Fails where `resolve` cannot close it: a projection it refuses leaves
  /// the call spelling a type it cannot make concrete.
  static FailureOr<CallSubstitution>
  forCall(SpecializationMap specialization, TypeRange operandTypes,
          TypeRange resultTypes, FunctionType formalTy,
          ProjectionResolver resolve);

  const SpecializationMap &getSpecialization() const { return specialization; }

  // A projection binding can rewrite a spelling into one that spells another
  // projection, so the ground half of this map is chased until it settles. Its
  // keys name ground spellings, so no chain through them reaches a key from its
  // own value; the parameter bindings ride along under keys nothing the chase
  // mints spells again.
  Type apply(Type ty) const {
    return applySubstitutionToFixedPoint(
        specialization,
        [&](Type key) -> std::optional<Type> {
          auto projection = dyn_cast<ProjectionType>(key);
          return projection ? projectionBindings.lookup(projection)
                            : std::nullopt;
        },
        ty, ClaimPredicates::Substituted);
  }

  const ProjectionBindings &getProjectionBindings() const {
    return projectionBindings;
  }

private:
  explicit CallSubstitution(SpecializationMap specialization)
      : specialization(std::move(specialization)) {}

  void discoverProjectionBindings(TypeRange types, ProjectionResolver resolve,
                                  bool &declined);

  SpecializationMap specialization;
  ProjectionBindings projectionBindings;
};

// Whether any occurrence of NeedleType is reachable in `ty`. An equality
// claim's endpoints are ordinary sub-elements, so the structural walk reaches a
// needle standing inside one.
template<class NeedleType> bool containsType(Type ty) {
  return ty.walk([](Type sub) {
             return isa<NeedleType>(sub) ? WalkResult::interrupt()
                                         : WalkResult::advance();
           }).wasInterrupted();
}

inline bool isMonomorphicType(Type ty) {
  return !isPolymorphicType(ty);
}

// A type is "ground" when it contains no PolymorphicTypeInterface nodes at all —
// no poly vars, no projections, no claims. Unlike
// isMonomorphicType, which asks whether any participant *reports* as polymorphic,
// isGroundType asks whether any participant *exists*. A monomorphic projection
// like !trait.proj<@Foo[i64], "Bar"> is monomorphic (no poly vars) but not
// ground (the projection still needs resolution).
inline bool isGroundType(Type root) {
  bool found = false;
  root.walk([&](Type sub) -> WalkResult {
    if (isa<PolymorphicTypeInterface>(sub)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !found;
}

/// Refine implementation for ops whose `inferReturnTypes` refuses to mint
/// fresh PolyTypes: when inference fails, there is nothing to refine and
/// verification accepts the declared result types as-is (an opaque
/// polymorphic input determines nothing about the result). When inference
/// succeeds, the default compatibility check applies. Ops declare
/// InferTypeOpInterface with ["refineReturnTypes"] and delegate here.
template <typename ConcreteOp>
LogicalResult refineUnlessUnmintable(MLIRContext *ctx,
                                     std::optional<Location> location,
                                     ValueRange operands, DictionaryAttr attrs,
                                     OpaqueProperties properties,
                                     RegionRange regions,
                                     SmallVectorImpl<Type> &returnTypes) {
  SmallVector<Type, 4> inferred;
  if (failed(ConcreteOp::inferReturnTypes(ctx, location, operands, attrs,
                                          properties, regions, inferred)))
    return success();
  if (!ConcreteOp::isCompatibleReturnTypes(inferred, returnTypes))
    return emitOptionalError(
        location, "'", ConcreteOp::getOperationName(), "' op inferred type(s) ",
        inferred, " are incompatible with return type(s) of operation ",
        returnTypes);
  return success();
}

// returns true iff every PolymorphicTypeInterface inside `root` is polymorphic,
// and at least one such participant exists
inline bool isPurelyPolymorphicType(Type root) {
  bool sawPoly = false;

  // fast path: if the root itself is a PolymorphicTypeInterface
  // call its predicate
  if (auto p = dyn_cast<PolymorphicTypeInterface>(root)) {
    if (p.isMonomorphic())
      return false; // root participates and is monomorphic -> not purely polymorphic
    sawPoly = true; // root participates and is polymorphic
  }

  // otherwise, walk the type and check every participating subtype
  bool allParticipatingArePoly = true;
  root.walk([&](Type sub) -> WalkResult {
    // skip the root to avoid infinite recursion
    if (sub == root) return WalkResult::advance();

    if (auto p = dyn_cast<PolymorphicTypeInterface>(sub)) {
      if (p.isPolymorphic()) {
        sawPoly = true;
        return WalkResult::advance();
      }
      // found a participant, monomorphic subtype -> fail
      allParticipatingArePoly = false;
      return WalkResult::interrupt();
    }

    return WalkResult::advance(); // non-participating types are ignored by design
  });

  // must have seen at least one one polymorphic participant, and none that are monomorphic
  return allParticipatingArePoly && sawPoly;
}

inline Type applySubstitution(const SpecializationMap &specialization,
                              OtherBindings others, Type root,
                              ClaimPredicates claims) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<std::pair<Type, WalkResult>> {
    if (auto generic = dyn_cast<GenericTypeInterface>(t)) {
      // GenericTypeInterface types own generic specialization entirely;
      // don't recurse into their result.
      Type specialized = generic.specializeWith(specialization);
      // A generic type that cannot spell its specialized form yields no type at
      // all. Stamping that into the enclosing type would leave a hole in it, so
      // the type stands as written and whatever consumes it next is what
      // reports that it never resolved.
      if (!specialized)
        specialized = t;
      return std::make_pair(specialized, WalkResult::skip());
    }

    // Otherwise, check the bindings of non-generic keys such as projections
    // and evidence claims.
    if (others)
      if (std::optional<Type> value = others(t))
        return std::make_pair(*value, WalkResult::advance());

    return std::nullopt;
  });

  // Move the equality endpoints every replacer holds as a leaf -- and under
  // `VariablesAlone` an application's arguments too -- applying the variable
  // bindings alone: such a predicate receives variable bindings, never a
  // projection or evidence binding resolved inside it (`respellClaimPredicate`).
  replacer.addReplacement(
      [&](ClaimType claim) -> std::optional<std::pair<Type, WalkResult>> {
    if (claim.isApplication() && claims == ClaimPredicates::Substituted)
      return std::nullopt;
    return respellClaimPredicate(claim, [&](Type t) {
      return applySubstitution(specialization, nullptr, t,
                               ClaimPredicates::Substituted);
    });
  });

  return replacer.replace(root);
}

/// The pass budget the substitution fixed point spends before it gives up.
///
/// The chase settles in as many passes as the longest chain of keys it binds
/// through -- a small number, because the keys it is for are ground spellings: a
/// resolved projection and a proven claim each name a type, and naming one can
/// expose another, but nothing along such a chain is reached from its own value.
/// A key that is (a type variable bound to a spelling mentioning that same
/// variable) grows one level per pass and never settles, which is why parameter
/// bindings are stamped once instead of chased; this bound stops any growth a
/// caller still lets through well before it exhausts the stack in the structural
/// rewrite, and the depth it reaches stays walkable.
constexpr unsigned kSubstitutionFixedPointMaxPasses = 256;

/// The depth limit on the chain of instantiations or obligations that reaches
/// the work in hand.
///
/// A template that instantiates itself at a larger type, and an impl whose
/// where-clause demands its own trait at a larger type, both make progress at
/// every step: each step is a new declaration instance or a new application, so
/// no cycle guard sees a repeat and the recursion runs until the machine stops
/// it. A bound on how deep the chain runs is what tells such a chain from a
/// finite one. Rust bounds the same two recursions the same way, at the same
/// default (`recursion_limit`, 128). A projection whose binding spells another
/// makes the same kind of progress, so a chain of steps resolving a projection
/// stands under the same bound, counted as rustc's normalizer counts it: a
/// projection a step's answer spells stands one deeper than the projection
/// that step resolved, and one written in another's argument stands as deep as
/// the projection it is written in (`ProjectionNormalizer`). It bounds the
/// steps one reading takes along one chain, not a chain's length: an element
/// already read is answered from what it read to, at no further depth. The
/// steps one reading takes in all are bounded by the step budget
/// (`kProjectionStepBudget`).
constexpr unsigned kInstantiationDepthLimit = 128;

/// How many steps one reading of a spelling's projections may take
/// (`ProjectionNormalizer`): a step is a binding a projection resolves to,
/// read in its place. The depth limit bounds the steps a reading takes along
/// one chain; this bounds the steps it takes in all, which a binding that
/// branches to a finite depth doubles at every level while no chain nears the
/// depth limit. It counts steps, not the types a normal form holds, because
/// the steps are the reading's work and what they spell need not grow with
/// them: a branching whose leaves coincide takes 2^(d+1) steps to a normal
/// form of d+1 distinct types, and a binding spelling its argument twice,
/// nested d deep, spells a tree of 2^d leaves in d steps. rustc bounds
/// normalization by its recursion limit alone, and takes a finite
/// branching's exponential time; this budget is the dialect's own, valued by
/// the time a reading takes to meet it, about a second.
constexpr unsigned kProjectionStepBudget = 1u << 18;

/// How many frames from each end of a chain a refusal names. A chain at the
/// depth limit is a hundred-odd frames of the same shape; its ends say where it
/// started and what it grew into, and the frames between them say nothing more.
constexpr size_t kChainEndsNamed = 3;

/// Attaches the ends of `chain` to `diagnostic`, one note per frame through
/// `name`, with a note standing in for the frames between them.
template <typename FrameT>
void nameChainEnds(InFlightDiagnostic &diagnostic, ArrayRef<FrameT> chain,
                   llvm::function_ref<void(InFlightDiagnostic &, FrameT)> name) {
  if (chain.size() <= 2 * kChainEndsNamed) {
    for (const FrameT &frame : chain)
      name(diagnostic, frame);
    return;
  }
  for (const FrameT &frame : chain.take_front(kChainEndsNamed))
    name(diagnostic, frame);
  diagnostic.attachNote() << "... " << chain.size() - 2 * kChainEndsNamed
                          << " more frame(s) elided";
  for (const FrameT &frame : chain.take_back(kChainEndsNamed))
    name(diagnostic, frame);
}

/// One step of an obligation chain: the application asked about, and the proof
/// that states what stands below it where a proof states it.
///
/// Impl selection descends a candidate's where clause, which no proof mediates,
/// and leaves `proof` null. A proof derivation descends the subproofs a given
/// list names, and carries that symbol so a refusal can say which proof put the
/// next step on the chain.
struct ObligationFrame {
  TraitApplicationAttr application;
  SymbolRefAttr proof;
};

/// Fails where an obligation chain has reached the depth limit: `chain`, the
/// frames an obligation walk is part-way through, outermost first, under a
/// derivation that stands `height` frames itself.
///
/// Every frame on the chain can be a distinct application, so the cycle guard
/// never fires on a chain whose obligations keep growing the type they ask
/// about. The chain's length is what stops it. Every frame counts against the
/// bound, whichever trait it names: a frame is a recursion the walk is standing
/// in, and a chain that alternates traits stands as deep as one that repeats a
/// single trait. Impl selection deriving a candidate's where clause and a proof
/// derivation descending its subproofs count frames the same way, so one
/// obligation chain has one bound. Where an answer read back carries the
/// height of its derivation, the chain reaches as deep as that derivation
/// reached under it.
LogicalResult checkObligationChainDepth(ArrayRef<ObligationFrame> chain,
                                        unsigned height = 1);

/// Names at `anchor` the obligation chain `chain` reaching `app` past the
/// depth limit (`checkObligationChainDepth`): the chain is what says where the
/// growth came from, so its ends are named.
void emitObligationOverflow(Location anchor, TraitApplicationAttr app,
                            ArrayRef<ObligationFrame> chain,
                            unsigned height = 1);

/// Applies `subst` repeatedly until it reaches a fixed point, so the returned
/// type carries no component that `subst` would still rewrite. The fixed
/// point is over `subst` alone; a projection whose base grounds under the
/// substitution stays a (now-resolvable) projection for the resolution
/// patterns.
///
/// For ground keys alone -- a resolved projection, a proven claim -- where one
/// rewrite exposes another. A map keyed by a declaration's parameters is
/// stamped with `instantiate` instead: chasing one re-reads what a parameter
/// stood for as though it were the declaration's own spelling, so a parameter
/// whose argument mentions that parameter grows one level per pass. What such a
/// map hands back here is the partial the budget stopped at, spelled as
/// written, which every comparison downstream declines on.
inline Type applySubstitutionToFixedPoint(
    const SpecializationMap &specialization, OtherBindings others, Type ty,
    ClaimPredicates claims) {
  Type cur = ty;
  for (unsigned pass = 0; pass != kSubstitutionFixedPointMaxPasses; ++pass) {
    Type next = applySubstitution(specialization, others, cur, claims);
    if (!next || next == cur)
      break;
    cur = next;
  }
  return cur;
}


/// The classes a set of type equalities carves out of the types they mention:
/// a union-find over interned types. An equality is not a directed rule --
/// `A = B` and `B = A` say one thing -- and a class is read only for whether
/// two members share it, so no member heads a class by any choice a verdict
/// could observe.
class TypeEquivalence {
public:
  unsigned size() const { return terms.size(); }

  /// The index this structure knows `t` by, interning it if it is new.
  unsigned intern(Type t);

  /// The index of the member the class `id` falls in is represented by.
  unsigned findCanonical(unsigned id);

  /// Joins the classes of `a` and `b`.
  void unite(unsigned a, unsigned b);

  Type termAt(unsigned id) const { return terms[id]; }

private:
  llvm::DenseMap<Type, unsigned> ids;
  SmallVector<Type> terms;
  SmallVector<unsigned> parent;
};

// this walks an Attribute and looks for any occurrence of the given NeedleType
template<class NeedleType> bool containsType(Attribute attr) {
  bool found = false;
  attr.walk([&](Attribute sub) {
    if (auto ta = dyn_cast<TypeAttr>(sub)) {
      if (containsType<NeedleType>(ta.getValue()))
        found = true;
    }
  });
  return found;
}

// this walks an Operation and looks for any occurrence of the given NeedleType
// note that this search does not recurse into child operations
template<class NeedleType> bool opMentionsType(Operation *op) {
  // inspect operands
  for (Type t : op->getOperandTypes())
    if (containsType<NeedleType>(t)) return true;

  // inspect result types
  for (Type t : op->getResultTypes())
    if (containsType<NeedleType>(t)) return true;

  // inspect block arguments
  for (Region& r : op->getRegions())
    for (Block& b : r)
      for (Value arg : b.getArguments())
        if (containsType<NeedleType>(arg.getType()))
          return true;

  // inspect attributes
  for (NamedAttribute attr : op->getAttrs())
    if (containsType<NeedleType>(attr.getValue()))
      return true;

  return false;
}

/// Collects distinct generic types appearing anywhere in `ty`, in the order a
/// pre-order walk first meets them. The walk visits each distinct part once,
/// through the attributes a claim or a projection stores its trait application
/// and endpoints in as through any other part.
SmallVector<GenericTypeInterface, 4> getGenericTypesIn(Type ty);

/// The declaration parameter `ty` is an occurrence of, or null when it is none.
///
/// A type answers for itself through `getParameterAtom`; the parameter is the
/// label that answer is. A label is an occurrence of itself, and a
/// kind-constraining wrapper of the label it wraps; a type whose atom is no
/// label is no occurrence of a parameter.
inline GenericTypeInterface getParameterOccurrence(Type ty) {
  auto generic = dyn_cast<GenericTypeInterface>(ty);
  if (!generic)
    return {};
  if (auto label = dyn_cast<PolyType>(generic.getParameterAtom()))
    return cast<GenericTypeInterface>(Type(label));
  return {};
}

template <typename RangeT>
SpecializationMap SpecializationMap::fromPositions(RangeT &&arguments) {
  SpecializationMap result;
  for (Type argument : arguments)
    result.arguments.push_back(argument);
  return result;
}

/// The parameter count of a declaration whose header and where clause spell
/// `spellings`: one past the largest label they spell. A declaration's labels
/// are its parameters' positions (`verifyDenseLabels`), so this is the count
/// without a reading of the order the parameters first occur in.
unsigned getLabelBound(ArrayRef<Type> spellings);

/// The number of parameters `spellings` bind at labels `first` and past.
/// Fails where those labels skip a position: a declaration's parameters are
/// labelled 0, 1, ..., and a member's own continue its declaration's without a
/// gap.
FailureOr<unsigned> countDenseLabelsFrom(ArrayRef<Type> spellings,
                                         unsigned first);

/// The first label from `first` up to the largest `spellings` spell that none
/// of them spells; none where they spell those labels without a gap.
std::optional<unsigned> findSkippedLabel(ArrayRef<Type> spellings,
                                         unsigned first);

/// The type parameters `ty` binds, in first-occurrence order.
///
/// The distinct parameters the generics `ty` spells are occurrences of, each
/// counted once at the position it first appears: the parameter list of a
/// declaration that states none beyond the labels it spells, a function's
/// signature.
inline SmallVector<GenericTypeInterface, 4> getTypeParametersIn(Type ty) {
  SmallVector<GenericTypeInterface, 4> result;
  DenseSet<Type> seen;
  for (GenericTypeInterface generic : getGenericTypesIn(ty)) {
    GenericTypeInterface parameter = getParameterOccurrence(Type(generic));
    if (parameter && seen.insert(Type(parameter)).second)
      result.push_back(parameter);
  }
  return result;
}

/// The first `!trait.poly` label no type standing in `op`'s declaration spells.
///
/// A label names a position in the declaration that binds it, so a rewrite that
/// introduces a variable into a declaration already written -- the state of a
/// fold body, the intermediate tuple a flat map is cut into -- must not spell a
/// label that declaration, or any body already inside it, binds: a substitution
/// over either is keyed by those labels and would rewrite the new variable along
/// with them. Reading the labels standing in the declaration and taking the next
/// one is what keeps the new variable the rewrite's own.
///
/// The declaration is the outermost operation below the enclosing module that
/// `op` stands in -- the function, trait or impl whose parameters a substitution
/// is keyed by -- or `op` itself when it stands in none.
///
/// A declaration built from nothing needs none of this: its own parameters are
/// labelled 0, 1, ... by their position in its header, and its body is read
/// against that header alone.
unsigned firstUnusedPolyLabel(Operation *op);

/// The established context a comparison reads both sides through: an impl's own
/// bindings and premises inside a verifier, impl selection inside the stage. A caller with no context passes none, which leaves both sides spelled
/// as written. Failure means the rewrite has no normal form.
using Normalizer = llvm::function_ref<FailureOr<Type>(Type)>;

/// One optional type argument per parameter of a declaration, by label: slot
/// `i` is the argument of `!trait.poly<i>`, for the labels below the bound the
/// declaration spells (`getLabelBound`).
///
/// A slot is filled at the first position that determines it and compared for
/// identity at every later one, so a parameter occurring twice admits only an
/// actual spelling one type at both. Nothing here decides a match: filling a
/// slot wrongly can only make the rebuilt declaration differ from the actual,
/// which the comparison refuses.
class TypeArguments {
public:
  explicit TypeArguments(unsigned bound) : slots(bound) {}

  /// One past the largest label the declaration binds.
  unsigned getBound() const { return slots.size(); }

  /// Whether the declaration binds `parameter`.
  bool binds(GenericTypeInterface parameter) const {
    return labelOf(parameter) < slots.size();
  }

  /// Fills `parameter`'s slot with `value`, or checks that it already holds
  /// exactly that. Fails on a parameter this declaration does not bind and on a
  /// second, differing value.
  LogicalResult assign(GenericTypeInterface parameter, Type value,
                       llvm::function_ref<InFlightDiagnostic()> err);

  /// The argument `parameter` took, or nothing when it took none or when the
  /// declaration does not bind it.
  std::optional<Type> lookup(GenericTypeInterface parameter) const {
    return binds(parameter) ? slots[labelOf(parameter)] : std::nullopt;
  }

  /// The argument the parameter labelled `label` took, or nothing.
  std::optional<Type> at(unsigned label) const { return slots[label]; }

  /// Whether every parameter has an argument.
  bool complete() const {
    return llvm::all_of(slots, [](const std::optional<Type> &slot) {
      return slot.has_value();
    });
  }

  /// The substitution these arguments spell: each filled slot at its label. An
  /// empty slot leaves its parameter standing.
  SpecializationMap toSpecialization() const {
    SmallVector<Type> arguments;
    for (const std::optional<Type> &slot : slots)
      arguments.push_back(slot.value_or(Type()));
    return SpecializationMap::fromPositions(arguments);
  }

private:
  static unsigned labelOf(GenericTypeInterface parameter) {
    return cast<PolyType>(parameter.getParameterAtom()).getLabel();
  }

  SmallVector<std::optional<Type>, 4> slots;
};

/// `declared` with its parameters replaced by `args`, in one pass.
///
/// A term substituted in is never revisited, so a callee's parameter and a
/// caller's that happen to share a spelling stay apart and a substitution that
/// maps a parameter into a type mentioning it terminates instead of growing.
inline Type instantiate(Type declared, const SpecializationMap &args) {
  return args.apply(declared);
}

/// The unproven claim stating `predicate` -- a trait application or a type
/// equality -- with its parameters replaced by `args`. A substitution rewrites
/// the type arguments a claim carries and never the claim itself, so the
/// result is a claim.
inline ClaimType instantiatePredicate(Attribute predicate,
                                      const SpecializationMap &args) {
  return cast<ClaimType>(instantiate(
      Type(ClaimType::get(predicate.getContext(), predicate, nullptr)), args));
}

/// Reads `actual` for the arguments `formal`'s parameters take, filling `args`.
///
/// The walk runs in lockstep. A formal parameter occurrence takes the actual
/// subterm standing opposite it; the actual side is never read as a pattern, so
/// nothing it spells is narrowed to fit. A formal claim recurses through its
/// predicate by position, blind to the proof either side carries. Every other
/// node recurses only where the two sides carry the same constructor, the same
/// attributes and the same child count; where they diverge the reading stops,
/// because it runs before either side is normalized and has nothing to learn
/// there.
///
/// A projection is not injective, so nothing is read out of the position it
/// stands in; its arguments are read only against the same projection on the
/// actual side, and only once every position outside a projection has had its
/// say. A parameter read twice at two spellings keeps the first, for the same
/// reason both readings are safe: the two may yet be one type. So this cannot
/// refuse anything, which is what makes `verifyEqualAfterInstantiation` the
/// only verdict.
void extractTypeArguments(Type formal, Type actual, TypeArguments &args);

/// Whether `formal` instantiated at `args` is `actual`.
///
/// Both sides are stripped of the proofs their claims carry -- comparison is
/// modulo the proof, permanently -- and read through `normalize`, so two
/// spellings the caller's established context makes one compare equal. The
/// verdict is then identity on interned types: a parameter no argument filled
/// stands, and a projection the context cannot reduce is equal to itself alone.
LogicalResult verifyEqualAfterInstantiation(
    Type formal, const SpecializationMap &args, Type actual,
    Normalizer normalize, llvm::function_ref<InFlightDiagnostic()> err);

/// The arguments carrying `formal` to `actual`, where the declaration `formal`
/// comes from binds the labels below `bound`: read them off the actual, then
/// check the instantiated declaration is the actual.
FailureOr<SpecializationMap> matchDeclaration(
    unsigned bound, Type formal, Type actual, Normalizer normalize,
    llvm::function_ref<InFlightDiagnostic()> err);

/// Visits the sites of `root` at which instantiation can owe work: every
/// sub-type except the interior of a claim.
///
/// A claim's predicate holds a proposition, not work. An equality's endpoints
/// are a term the equation relates, discharged when the equality settles; an
/// application's arguments are the spelling its proof is minted at, which is
/// never respelled (`respellClaimPredicate`), so a projection standing in one
/// is the proof's to bridge in its own body and a claim standing in one is
/// proven before the claim around it. A scan that judges obligations sees the
/// claim itself and never what it states. Every such scan reads this walk, so
/// the rule is stated once.
void walkObligationSites(Type root, llvm::function_ref<void(Type)> visit);

/// Whether `site`, a sub-type `walkObligationSites` visits, is an obligation
/// instantiation has not discharged: an unproven monomorphic application claim,
/// or a ground projection, whose base is concrete and so resolves in place.
bool isUndischargedObligation(Type site);

/// Whether `root` carries an undischarged obligation at one of its sites.
///
/// A step that reads a spelling and cannot revisit what it read asks this
/// first, and waits while it holds: a name mangled, or a signature compared,
/// while an obligation still stands is read off a spelling the module no longer
/// has once the obligation is settled. A proven claim is no obligation, so a
/// type carrying one never holds a step back.
///
/// Defined out of line so that a dialect asking it links one symbol rather than
/// the type identities this dialect's own library carries.
bool carriesUndischargedObligation(Type root);

/// How many requirements `claim` carries: the requirements of the trait it
/// applies, plus -- when `claim` is proven by a proof -- the where entries of the
/// impl that proof derives it from. An equality claim applies no trait, so it
/// requires nothing.
FailureOr<uint64_t> getClaimRequirementCount(
    ClaimType claim,
    ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

/// The requirement `claim` carries at `index`, in the one order a projection
/// indexes them by: the trait's requirements, its result signature, then --
/// when `claim` is proven by a proof -- the where entries of the impl that
/// proof's derive cites, the order its operands supply them in. The requirement
/// is instantiated at the claim's arguments, a trait requirement of a proven
/// claim read through the cited impl's own bindings as its return is spelled.
/// A where entry of a proven claim carries the proof the derive's operand there
/// names; a trait requirement carries none here, its evidence being the impl's
/// return operand, which the stage inlines where a projection reads it
/// (`ProjectOp::inlineEvidence`). An equality requirement never carries a
/// provider. This is the one reading a `trait.project` hop is checked against,
/// and it reads no declaration's body. Refuses an index past the last
/// requirement.
FailureOr<ClaimType> getClaimRequirementAt(
    ClaimType claim,
    ModuleOp module,
    uint64_t index,
    llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);

/// Refuses `claim` where it states an equality one of whose endpoints holds a
/// claim, proven or not, at its root or inside another type, naming the
/// endpoint through `err`: an equality relates types, and a claim is a value's
/// evidence (`TypeEqualityAttr`).
LogicalResult verifyEndpointsRelateTypes(
    ClaimType claim, llvm::function_ref<InFlightDiagnostic()> err);

/// Whether the evidence `proven` names is evidence for its application: the
/// declaration the cited symbol holds -- a proof's proven application, which
/// `ProofOp::verify` holds ground, or an unconditional impl's header -- is that
/// application, by interned identity. Every writer witnesses a proof at the
/// application it proves and respells it by a coercion whose result names no
/// proof, so a citation is one comparison, and every proven claim type a
/// verified module holds passes it (`ClaimType::verifySymbolUses`). Nothing is
/// read inside a cited proof: a proof op's body decides its own premises and
/// citations.
LogicalResult verifyCitation(ClaimType proven, ModuleOp module,
                             llvm::function_ref<InFlightDiagnostic()> err);

/// The module that anchors symbol lookups for `anchor`: the operation itself
/// when it is the module, otherwise its enclosing module (null if it has none).
/// A symbol-use verifier reached through an attribute or type interface recovers
/// its lookup scope this way.
ModuleOp getAnchorModule(Operation *anchor);

/// The bound a reading of a spelling's projections meets before it reaches a
/// normal form (`ProjectionNormalizer`).
enum class NormalizationBound {
  /// A chain of steps that does not end within the depth limit
  /// (`kInstantiationDepthLimit`): a binding that grows, at its root or beneath
  /// a constructor, spells a projection one step deeper at every step, and one
  /// whose reading reaches a projection still being read, by its own binding
  /// or through another's, has no end at all.
  Depth,
  /// A reading that would take more steps than the step budget
  /// (`kProjectionStepBudget`): a binding that branches to a finite depth
  /// doubles the steps at every level, whether its leaves are distinct or
  /// coincide. (One that branches without end meets the depth limit first,
  /// along its first branch.)
  Steps,
};

/// What reading a spelling's projections reached: the normal form, or the
/// partial normal form standing where a bound was met, with that bound.
struct Normalized {
  Type type;
  /// The bound met; none where `type` is the normal form.
  std::optional<NormalizationBound> unsettled;
};

/// Reads a spelling's projections to their normal form in one walk, as rustc's
/// normalizer folds a type and GHC's rewriter rewrites a type family
/// application. A projection's arguments -- its head application's and its own
/// associated-type arguments -- are read first, at the depth the projection
/// stands at; a projection `reads` admits is then offered to `hop`, which
/// answers the binding it resolves to or declines with nullopt; and the
/// answer, which may spell projections of its own (an impl whose associated
/// type forwards through its own type parameter, `type Element = B::Element`,
/// binds one), is read one step deeper before it stands where the projection
/// stood. So a projection's depth is that of the step whose answer spelled it,
/// plus one, and a projection written in another's argument adds none.
///
/// A projection stands, as read, where `reads` does not admit it, where `hop`
/// declines it, and where an argument still holds a projection `reads` admits
/// -- one `hop` declined. An argument that is a claim, or holds one, is
/// judged where the step is constructed (`ProjectionResolution::get`), which
/// takes no step at it. The reading meets a bound where a projection would be
/// offered at the depth limit, where it reaches an element it is still reading
/// (whose normal form would then hold itself, so the chain never ends), or
/// where it would take a step past the step budget; it then reads nothing more
/// and hands back the partial normal form it reached. Each element is read
/// once and what it read to is kept, so a projection spelled twice is offered
/// once, and a spelling whose tree is exponential in its depth costs its
/// distinct types.
///
/// Every normalizer reads through this, so the spelling one hands back is the
/// spelling all of them do, which is what lets a demand and an impl's self
/// application be compared for equality at all.
class ProjectionNormalizer {
public:
  ProjectionNormalizer(std::function<bool(ProjectionType)> reads,
                       std::function<std::optional<Type>(ProjectionType)> hop);
  ProjectionNormalizer(const ProjectionNormalizer &) = delete;
  ProjectionNormalizer &operator=(const ProjectionNormalizer &) = delete;

  /// `ty` read to its normal form, or the partial normal form standing where
  /// a bound was met, with that bound: this reading's, or an earlier one's
  /// through this normalizer, after which nothing is read. A bound met is no
  /// answer about the type; the caller, which owns the diagnostic, decides
  /// what it means and names the type that would not settle.
  Normalized normalize(Type ty);

private:
  /// Reads `projection` at `depth` (above), and what it reads to.
  std::pair<Type, WalkResult> read(ProjectionType projection);

  /// Meets a bound where `normal`, the normal form of an answer, holds a
  /// projection still being read. The replacer answers an element reached
  /// again while it is being read with that element as written; it holds a
  /// projection being read, and nothing reading it takes it apart -- a
  /// projection holding it in an argument stands -- so it stands in the
  /// normal form of the answer being read when it was reached: a chain that
  /// returns to where it is reading has no end.
  ///
  /// Each type is judged once per normalizer: `walker` keeps what it has
  /// walked across answers and readings. A type judged clean stays clean,
  /// because every projection standing in a normal form has been through the
  /// replacer. Its reading had either finished when that normal form was
  /// judged -- and an element read is answered from what it read to and never
  /// read again, so it is never again being read -- or was still in
  /// progress, and the judgment met the bound, after which nothing is read or
  /// judged.
  void judge(Type normal);

  std::function<bool(ProjectionType)> reads;
  std::function<std::optional<Type>(ProjectionType)> hop;
  /// Keeps what each element read to, so each is read once.
  AttrTypeReplacer replacer;
  /// Finds a projection still being read; keeps what it has walked, so each
  /// type is judged once.
  AttrTypeWalker walker;
  /// The projections whose reading is in progress.
  DenseSet<Type> reading;
  /// How many answers are being read beneath the element in hand.
  unsigned depth = 0;
  /// How many steps the reading in hand has taken.
  unsigned steps = 0;
  std::optional<NormalizationBound> met;
};

/// The symbol suffix `_h` followed by sixteen hex digits of `input`'s hash.
std::string hashToSuffix(StringRef input);

std::string generateMangledNameSuffixFor(TypeRange typeArgs);

std::string applySubstitutionAndGenerateMangledNameSuffix(
    const SpecializationMap &subst, ArrayRef<GenericTypeInterface> typeParams);

} // end mlir::trait
