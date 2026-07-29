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
// concrete argument into a projection spelling can mint a ground redex the
// fixed point alone does not close; when `module` is supplied the replacer
// resolves those redexes by module-visible impl lookup, so a specialized
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
      result = resolveGroundProjectionsByLookup(result, module,
                                                DemandOrigin::MonomorphStampOut);

    // check that the result changed
    return (result != t) ? std::optional<Type>(result) : std::nullopt;
  });
  return replacer;
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

  // make a type replacer that also resolves the ground projection redexes this
  // substitution mints, so the specialized instance is stamped in normal form
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(
      substitution, polymorph->getParentOfType<ModuleOp>());

  // replace the polymorphic function type
  auto oldFunctionType = polymorph.getFunctionType();
  auto newFunctionType = llvm::cast<FunctionType>(replacer.replace(oldFunctionType));

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

  // make a type replacer that also resolves the ground projection redexes this
  // substitution mints, so the specialized region is stamped in normal form
  ModuleOp module =
      polymorph.getParentOp() ? polymorph.getParentOp()->getParentOfType<ModuleOp>()
                              : ModuleOp();
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst, module);

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph,
                                 monomorph,
                                 mapping,
                                 replacer);

  // A detached region has no module, so the replacer above had no lookup to
  // resolve with. Whether that matters is a question about the clone, not about
  // the source: substituting a concrete argument into a symbolic projection is
  // what makes it monomorphic, so a projection invisible in the polymorph can
  // be a ground redex here. Count them, over the same result and block-argument
  // types the stage's own leftover-projection sweep walks.
  //
  // No caller reaches this today: every one of them passes a region whose
  // parent op is in a module. The walk is a whole-region traversal, so it is
  // guarded rather than paid for on a path a detached-region caller would make
  // hot on the day one appears.
  if (!module && DemandLedger::areObservationsEnabled()) {
    auto count = [](Type root) {
      root.walk([](Type sub) {
        auto proj = dyn_cast<ProjectionType>(sub);
        if (proj && isMonomorphicType(proj))
          countModulelessRegionProjection();
      });
    };
    for (Block &block : monomorph)
      for (BlockArgument arg : block.getArguments())
        count(arg.getType());
    monomorph.walk([&](Operation *op) {
      for (Type t : op->getResultTypes())
        count(t);
      for (Region &r : op->getRegions())
        for (Block &b : r)
          for (BlockArgument arg : b.getArguments())
            count(arg.getType());
    });
  }
}

} // end mlir::trait
