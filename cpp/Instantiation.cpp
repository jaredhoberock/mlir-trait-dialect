// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Instantiation.hpp"
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

AttrTypeReplacer makeTypeReplacerFromSubstitution(const DenseMap<Type,Type> &subst) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([=](Type t) -> std::optional<Type> {
    Type result = applySubstitutionToFixedPoint(subst, t);

    // check that the result changed
    return (result != t) ? std::optional<Type>(result) : std::nullopt;
  });
  return replacer;
}

func::FuncOp instantiatePolymorph(OpBuilder& builder,
                                  func::FuncOp polymorph,
                                  StringRef instanceName,
                                  const DenseMap<Type,Type> &substitution) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot instantiate external function");
    return nullptr;
  }

  Location loc = polymorph.getLoc();

  // make a type replacer
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(substitution);

  // replace the polymorphic function type
  auto oldFunctionType = polymorph.getFunctionType();
  auto newFunctionType = llvm::cast<FunctionType>(replacer.replace(oldFunctionType));

  // create the instance with the new type and instance name
  func::FuncOp instance = builder.create<func::FuncOp>(loc, instanceName, newFunctionType);

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

void instantiatePolymorphicRegion(OpBuilder& builder,
                                  Region& polymorph,
                                  Region& monomorph,
                                  const DenseMap<Type,Type> &subst) {
  assert(monomorph.empty() && "Region is not empty");

  // make a type replacer
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst);

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph,
                                 monomorph,
                                 mapping,
                                 replacer);
}

} // end mlir::trait
