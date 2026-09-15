// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/SetVector.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/CallInterfaces.h>

namespace mlir::trait {

/// Whether `attr` is a generic call's type_params array: the callee's own type
/// variables, named by identity. A substitution over the enclosing template binds
/// the template's variables, and the callee's variables are the same variables
/// only when the callee is a template the clone is cut from -- the enclosing
/// function calling itself, or an impl's method forwarding the same trait method
/// to another impl, whose call names the trait method's variables the outer
/// call's bindings are keyed by. Either way the array must name the callee's
/// variables after the clone as before it; the parallel type_args array is what
/// takes the substitution.
static bool namesCalleeTypeVariables(Operation &op, NamedAttribute attr) {
  if (auto call = dyn_cast<FuncCallOp>(&op))
    return attr.getName() == call.getTypeParamsAttrName();
  if (auto call = dyn_cast<MethodCallOp>(&op))
    return attr.getName() == call.getTypeParamsAttrName();
  return false;
}

/// Clone block and successor mappings, then substitute in region order.
/// Builder notifications admit the copied operations to rewrite listeners;
/// visiting block arguments first preserves transient projection demand order.
static void cloneRegionWithTypeReplacement(
    OpBuilder& builder,
    Region &oldRegion,
    Region &newRegion,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer) {
  if (oldRegion.empty())
    return;
  builder.cloneRegionBefore(oldRegion, newRegion, newRegion.end(), mapping);
  auto substituteRegion = [&](Region &region, auto &recurse) -> void {
    for (Block &block : region)
      for (BlockArgument arg : block.getArguments())
        arg.setType(typeReplacer.replace(arg.getType()));
    for (Block &block : region) {
      for (Operation &op : block) {
        for (Value result : op.getResults())
          result.setType(typeReplacer.replace(result.getType()));
        for (NamedAttribute attr : op.getAttrs())
          if (!namesCalleeTypeVariables(op, attr))
            op.setAttr(attr.getName(), typeReplacer.replace(attr.getValue()));
        for (Region &nested : op.getRegions())
          recurse(nested, recurse);
      }
    }
  };
  substituteRegion(newRegion, substituteRegion);
}

// A template clone -- one stamped with no module -- receives the bindings of a
// declaration's parameters alone, and each is stamped once: what a parameter
// stands for is a term of whoever supplied it, so reading that term again as
// though it were the declaration's own spelling would mistake a shared label for
// the same variable and grow a parameter bound over itself one level per pass.
// A monomorphic clone receives the closed call substitution, whose projection
// and evidence bindings expose one another, so those are chased until they
// settle. Substituting a concrete argument into a projection spelling can mint a
// ground projection no substitution entry closes; when `module` is supplied the
// replacer resolves those projections by module-visible impl lookup, so a
// specialized monomorph carries no ground projection that a unique
// module-visible impl resolves. Projections whose impl is generator-pending or
// whose application matches several candidates survive stamp-out unchanged, to
// be resolved once evidence exists.
AttrTypeReplacer makeTypeReplacerFromSubstitution(const DenseMap<Type,Type> &subst,
                                                  ModuleOp module) {
  // The seal keeps a bare equality -- a witness's stored evidence, a
  // where-clause predicate -- immutable under this rewrite; the clone rule
  // below is the one mover, and it reaches an equality only through the claim
  // that wraps it.
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  replacer.addReplacement(
      [=](Type t) -> std::optional<std::pair<Type, WalkResult>> {
    // A template's clone: the substitution is a structural rewrite of the whole
    // type already, so what it stamped in is the answer and the walk does not
    // re-enter it. Re-entering is what reads a parameter's argument as though it
    // were the declaration's own spelling again, which grows a parameter bound
    // over itself one level per visit.
    if (!module)
      return std::make_pair(applySubstitutionOnce(subst, t),
                            WalkResult::skip());

    Type result = applySubstitutionToFixedPoint(subst, t);
    result = resolveProjectionsByLookup(result, module,
                                        DemandOrigin::MonomorphStampOut,
                                        LookupScope::Ground);

    // A generic type owns its specialization entirely, so the substitution
    // reaches the parameter it stands for through `specializeWith` and never
    // through its sub-elements. A kinded occurrence such as `!tuple.poly<P>`
    // whose argument is not a tuple answers with no type, which leaves it
    // spelled as written; descending into it would rebuild the occurrence
    // around the argument, which is not a parameter at all.
    if (isa<GenericTypeInterface>(t))
      return std::make_pair(result, WalkResult::skip());

    // check that the result changed
    return (result != t)
               ? std::optional<std::pair<Type, WalkResult>>(
                     std::make_pair(result, WalkResult::advance()))
               : std::nullopt;
  });

  // The clone rule for equality evidence: the endpoints receive the variable
  // bindings alone, stamped once -- no projection or evidence binding, and no
  // module lookup, resolved inside them -- matching what the witness verifier
  // enforces: the current endpoints must be a single-substitution instance of
  // the witness's own equality, which a resolution would break. A witness's
  // stored equality is likewise NOT rewritten -- it is immutable evidence.
  llvm::DenseMap<Type, Type> variableBindings;
  for (auto [key, value] : subst)
    if (isa<GenericTypeInterface>(key))
      variableBindings.try_emplace(key, value);
  replacer.addReplacement(
      [variableBindings](ClaimType claim)
          -> std::optional<std::pair<Type, WalkResult>> {
    return respellEqualityEndpoints(claim, [&](Type t) {
      return applySubstitutionOnce(variableBindings, t);
    });
  });

  return replacer;
}

/// Whether the block a builder inserts into stands inside a trait, impl, or
/// proof, or a still-polymorphic function -- a template, whose clone carries no
/// projection binding, no evidence binding, and no module lookup, because its
/// spelling is resolved when the template is itself cloned for a concrete
/// instance.
static bool insertionStandsInsideTemplate(OpBuilder &builder) {
  Block *block = builder.getInsertionBlock();
  if (!block)
    return false;
  for (Operation *op = block->getParentOp(); op; op = op->getParentOp()) {
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return true;
    if (auto func = dyn_cast<func::FuncOp>(op))
      if (isPolymorphicType(Type(func.getFunctionType())))
        return true;
  }
  return false;
}

namespace {

/// The first mention of each type variable a substitution binds nothing for, in
/// the order the reading meets them, so each is refused once and at a site.
struct UnboundVariables {
  SmallVector<std::pair<Type, Operation *>> inOrder;
  DenseSet<Type> seen;

  /// Reads `root`, a type or an attribute, for the variables `bound` does not
  /// bind, recording a first mention at `at`.
  template <typename RootT>
  void read(RootT root, const SetVector<Type> &bound, Operation *at) {
    root.walk([&](Type sub) {
      for (GenericTypeInterface variable : getTypeParametersIn(sub)) {
        Type label(variable);
        if (bound.contains(label))
          continue;
        if (seen.insert(label).second)
          inOrder.emplace_back(label, at);
      }
    });
  }
};

void readRegion(Region &region, const SetVector<Type> &bound,
                UnboundVariables &unbound);

/// Reads `op` and whatever it holds against `bound`, the variables a
/// substitution has an argument for where `op` stands.
void readOp(Operation *op, const SetVector<Type> &bound,
            UnboundVariables &unbound) {
  for (Type type : op->getResultTypes())
    unbound.read(type, bound, op);
  for (NamedAttribute attribute : op->getAttrs())
    if (!namesCalleeTypeVariables(*op, attribute))
      unbound.read(attribute.getValue(), bound, op);

  // A callable region a scope reaches into is a lambda a dialect specializes
  // once per use -- a `tuple.map` body, once per element type -- so its own
  // signature supplies arguments for what it holds, on top of the ones the
  // substitution already binds. Every other region is interior to the scope
  // around it and binds nothing of its own.
  SetVector<Type> inside = bound;
  if (auto callable = dyn_cast<CallableOpInterface>(op)) {
    auto bindSignature = [&](ArrayRef<Type> types) {
      for (Type type : types)
        for (GenericTypeInterface variable : getTypeParametersIn(type))
          inside.insert(Type(variable));
    };
    bindSignature(callable.getArgumentTypes());
    bindSignature(callable.getResultTypes());
  }

  for (Region &region : op->getRegions())
    readRegion(region, inside, unbound);
}

void readRegion(Region &region, const SetVector<Type> &bound,
                UnboundVariables &unbound) {
  for (Block &block : region) {
    for (BlockArgument argument : block.getArguments())
      unbound.read(argument.getType(), bound, region.getParentOp());
    for (Operation &op : block)
      readOp(&op, bound, unbound);
  }
}

} // namespace

/// Refuses every type variable `polymorph`'s body spells that `substitution`
/// binds no argument for.
///
/// A clone is stamped out of a declaration under a substitution keyed by that
/// declaration's parameters, so a variable the substitution does not bind has
/// nothing to receive and rides into the clone spelled as written. The reading
/// visits exactly what the clone's substitution visits -- block argument types,
/// result types and attributes, region by region -- minus the array a generic
/// call spells its CALLEE's parameters in, which stands in the callee's scope.
static LogicalResult refuseUnboundVariables(
    func::FuncOp polymorph, const DenseMap<Type,Type> &substitution) {
  SetVector<Type> bound;
  for (auto [key, value] : substitution)
    if (isa<GenericTypeInterface>(key))
      for (GenericTypeInterface variable : getTypeParametersIn(key))
        bound.insert(Type(variable));

  // The signature is the declaration the substitution is keyed by, and the
  // clone's is monomorphic where this is asked: the judgment is about the body.
  UnboundVariables unbound;
  for (NamedAttribute attribute : polymorph->getAttrs())
    if (attribute.getName() != polymorph.getFunctionTypeAttrName())
      unbound.read(attribute.getValue(), bound, polymorph.getOperation());
  for (Region &region : polymorph->getRegions())
    readRegion(region, bound, unbound);

  for (auto [variable, at] : unbound.inOrder) {
    InFlightDiagnostic diagnostic =
        polymorph.emitError()
        << "type variable " << variable << " in the body of '@"
        << polymorph.getSymName()
        << "' is bound by no parameter of its declaration, so no instance can "
           "replace it";
    diagnostic.attachNote(at->getLoc()) << "mentioned here";
  }
  return success(unbound.inOrder.empty());
}

func::FuncOp specializePolymorph(OpBuilder& builder,
                                  func::FuncOp polymorph,
                                  StringRef instanceName,
                                  const DenseMap<Type,Type> &substitution) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot specialize external function");
    return nullptr;
  }

  Location loc = polymorph.getLoc();

  // A clone whose signature still spells a type variable under the generic-keyed
  // bindings alone, or that is inserted inside a trait, impl, or proof, is a
  // template: it is stamped under those bindings with no projection or evidence
  // binding and no module lookup, so a substitution-invariant verifier accepts
  // it as it accepts the source, and its spelling resolves when it is cloned for
  // a concrete instance. A monomorphic clone receives the full call substitution
  // and ground-projection normalization by module lookup.
  llvm::DenseMap<Type, Type> variableBindings;
  for (auto [key, value] : substitution)
    if (isa<GenericTypeInterface>(key))
      variableBindings.try_emplace(key, value);
  AttrTypeReplacer variableReplacer =
      makeTypeReplacerFromSubstitution(variableBindings, ModuleOp());

  auto oldFunctionType = polymorph.getFunctionType();
  auto substitutedType =
      llvm::cast<FunctionType>(variableReplacer.replace(oldFunctionType));

  bool cloneIsTemplate = isPolymorphicType(Type(substitutedType)) ||
                         insertionStandsInsideTemplate(builder);

  // A monomorphic clone receives an argument for every parameter its
  // declaration binds, so nothing it copies may spell one the substitution does
  // not bind: that spelling would stand in ground code as written, and the
  // declaration is where the missing parameter belongs. A template clone keeps
  // the parameters its own signature still spells, and receives its arguments
  // when it is itself cloned for a concrete instance.
  if (!cloneIsTemplate && failed(refuseUnboundVariables(polymorph, substitution)))
    return nullptr;

  AttrTypeReplacer fullReplacer = makeTypeReplacerFromSubstitution(
      substitution, polymorph->getParentOfType<ModuleOp>());
  AttrTypeReplacer &replacer = cloneIsTemplate ? variableReplacer : fullReplacer;

  auto newFunctionType =
      cloneIsTemplate
          ? substitutedType
          : llvm::cast<FunctionType>(replacer.replace(oldFunctionType));

  // create the instance with the new type and instance name
  func::FuncOp instance = func::FuncOp::create(builder, loc, instanceName, newFunctionType);

  // clone the polymorph's attributes with type replacement
  for (NamedAttribute attr : polymorph->getAttrs()) {
    StringRef n = attr.getName();

    // don't copy the polymorph's name or function type
    if (n == polymorph.getSymNameAttrName() ||
        n == polymorph.getFunctionTypeAttrName()) {
      continue;
    }

    instance->setAttr(attr.getName(), replacer.replace(attr.getValue()));
  }

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph.getBody(),
                                 instance.getBody(),
                                 mapping,
                                 replacer);

  return instance;
}

void specializePolymorphicRegion(OpBuilder& builder,
                                  Region& polymorph,
                                  Region& monomorph,
                                  const DenseMap<Type,Type> &subst) {
  assert(monomorph.empty() && "Region is not empty");

  // A region cloned into a template carries no module lookup: its projections
  // resolve when the template is cloned for a concrete instance, not here. A
  // region cloned into monomorphic code resolves its ground projections by
  // module-visible impl lookup, so the specialized region is stamped in normal
  // form.
  ModuleOp module =
      insertionStandsInsideTemplate(builder)
          ? ModuleOp()
          : (polymorph.getParentOp()
                 ? polymorph.getParentOp()->getParentOfType<ModuleOp>()
                 : ModuleOp());
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst, module);

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph,
                                 monomorph,
                                 mapping,
                                 replacer);
}

} // end mlir::trait
