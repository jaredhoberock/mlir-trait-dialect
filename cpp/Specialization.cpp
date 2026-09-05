// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "TraitTypes.hpp"
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>

namespace mlir::trait {

static void cloneRegionWithTypeReplacement(
    OpBuilder& builder,
    Region &oldRegion,
    Region &newRegion,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer);

static Operation *cloneOpWithTypeReplacement(
    OpBuilder &builder,
    Operation &oldOp,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer) {
  PatternRewriter::InsertionGuard guard(builder);

  OperationState state(oldOp.getLoc(), oldOp.getName());

  // remap operands
  for (Value operand : oldOp.getOperands())
    state.addOperands(mapping.lookupOrDefault(operand));

  // replace result types
  for (Type t : oldOp.getResultTypes())
    state.addTypes(typeReplacer.replace(t));

  // replace attributes
  for (NamedAttribute attr : oldOp.getAttrs()) {
    Attribute rewritten = typeReplacer.replace(attr.getValue());
    state.addAttribute(attr.getName(), rewritten);
  }

  // create empty regions in the new op
  for ([[maybe_unused]] Region &oldRegion : oldOp.getRegions()) {
    state.addRegion();
  }

  // create the operation *before* recursing into the old op's regions
  Operation *newOp = builder.create(state);

  // recursively clone regions
  for (auto [oldRegion, newRegion] : llvm::zip(oldOp.getRegions(), newOp->getRegions())) {
    cloneRegionWithTypeReplacement(builder, oldRegion, newRegion,
                                   mapping, typeReplacer);
  }

  // remap results
  for (auto [oldRes, newRes] : llvm::zip(oldOp.getResults(), newOp->getResults()))
    mapping.map(oldRes, newRes);

  return newOp;
}

static void cloneRegionWithTypeReplacement(
    OpBuilder& builder,
    Region &oldRegion,
    Region &newRegion,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer) {
  PatternRewriter::InsertionGuard guard(builder);

  // create blocks with replaced argument types
  for (Block &oldBlock : oldRegion.getBlocks()) {
    Block *newBlock = builder.createBlock(&newRegion);
    for (BlockArgument oldArg : oldBlock.getArguments()) {
      Type newType = typeReplacer.replace(oldArg.getType());
      BlockArgument newArg = newBlock->addArgument(newType, oldArg.getLoc());
      mapping.map(oldArg, newArg);
    }
  }

  // clone each operation in each new block
  auto& oldBlocks = oldRegion.getBlocks();
  auto& newBlocks = newRegion.getBlocks();
  for (auto [oldBlock, newBlock] : llvm::zip(oldBlocks, newBlocks)) {
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&newBlock);

    for (Operation &op : oldBlock) {
      cloneOpWithTypeReplacement(builder, op, mapping, typeReplacer);
    }
  }
}

// Every type this replacer stamps into a specialized clone is chased to the
// substitution's fixed point, so a specialized monomorph never carries a type
// that some remaining substitution entry would still rewrite. Substituting a
// concrete argument into a projection spelling can mint a ground projection the
// fixed point alone does not close; when `module` is supplied the replacer
// resolves those projections by module-visible impl lookup, so a specialized
// monomorph carries no ground projection that a unique module-visible impl
// resolves. Projections whose impl is generator-pending or
// whose application matches several candidates survive stamp-out unchanged, to
// be resolved once evidence exists.
AttrTypeReplacer makeTypeReplacerFromSubstitution(const DenseMap<Type,Type> &subst,
                                                  ModuleOp module) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([=](Type t) -> std::optional<Type> {
    Type result = applySubstitutionToFixedPoint(subst, t);
    if (module)
      result = resolveProjectionsByLookup(result, module,
                                          DemandOrigin::MonomorphStampOut,
                                          LookupScope::Ground);

    // check that the result changed
    return (result != t) ? std::optional<Type>(result) : std::nullopt;
  });

  // The clone rule for equality evidence: the endpoints receive the variable
  // bindings alone, to a fixed point -- no projection or evidence binding, and
  // no module lookup, resolved inside them -- matching what the witness verifier
  // enforces: the current endpoints must be a single-substitution instance of
  // the witness's own equality, which a resolution would break. A witness's
  // stored equality is likewise NOT rewritten -- it is immutable evidence.
  llvm::DenseMap<Type, Type> variableBindings;
  for (auto [key, value] : subst)
    if (isa<GenericTypeInterface>(key))
      variableBindings.try_emplace(key, value);
  replacer.addReplacement([variableBindings](ClaimType claim) -> std::optional<Type> {
    return respellEqualityEndpoints(claim, [&](Type t) {
      return applySubstitutionToFixedPoint(variableBindings, t);
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
