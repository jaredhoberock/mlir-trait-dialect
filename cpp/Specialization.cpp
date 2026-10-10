// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "SymbolLookup.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/SetVector.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>

namespace mlir::trait {

void cloneRegionStampedBefore(OpBuilder &builder, Region &source, Region &dest,
                              Region::iterator before, IRMapping &mapping,
                              AttrTypeReplacer &typeReplacer,
                              AttrTypeReplacer &spellingReplacer) {
  if (source.empty())
    return;
  // The clones stand from the block after the one preceding `before` up to
  // `before`.
  Block *preceding = before == dest.begin() ? nullptr : &*std::prev(before);
  builder.cloneRegionBefore(source, dest, before, mapping);
  // Each region's block arguments are stamped before its ops, which keeps the
  // order transient projection demands are raised in.
  auto substitute = [&](auto blocks, auto &recurse) -> void {
    for (Block &block : blocks)
      for (BlockArgument arg : block.getArguments())
        arg.setType(typeReplacer.replace(arg.getType()));
    for (Block &block : blocks) {
      for (Operation &op : block) {
        auto evidenceCall = dyn_cast<MethodCallOp>(&op);
        AttrTypeReplacer &resultReplacer =
            evidenceCall && evidenceCall.computesEvidence() ? spellingReplacer
                                                            : typeReplacer;
        for (Value result : op.getResults())
          result.setType(resultReplacer.replace(result.getType()));
        for (NamedAttribute attr : op.getAttrs())
          op.setAttr(attr.getName(), typeReplacer.replace(attr.getValue()));
        for (Region &nested : op.getRegions())
          recurse(llvm::make_range(nested.begin(), nested.end()), recurse);
      }
    }
  };
  substitute(llvm::make_range(preceding ? std::next(preceding->getIterator())
                                        : dest.begin(),
                              before),
             substitute);
}

/// Clones `oldRegion` into the end of `newRegion`, stamped
/// (`cloneRegionStampedBefore`).
static void cloneRegionWithTypeReplacement(
    OpBuilder& builder,
    Region &oldRegion,
    Region &newRegion,
    IRMapping &mapping,
    AttrTypeReplacer &typeReplacer,
    AttrTypeReplacer &spellingReplacer) {
  cloneRegionStampedBefore(builder, oldRegion, newRegion, newRegion.end(),
                           mapping, typeReplacer, spellingReplacer);
}

// A template clone receives the bindings of a declaration's parameters alone,
// and each is stamped once: what a parameter stands for is a term of whoever
// supplied it, so reading that term again as though it were the declaration's
// own spelling would mistake a shared label for the same variable and grow a
// parameter bound over itself one level per pass. An instance's clone receives
// the closed call substitution, whose projection and evidence bindings expose
// one another, so those are chased until they settle. Substituting a concrete
// argument into a projection spelling can mint a ground projection no
// substitution entry closes; it stays spelled, for the op carrying it to ask
// impl selection about.
AttrTypeReplacer makeTypeReplacerFromSubstitution(
    const SpecializationMap &variableBindings, CloneKind kind,
    const ProjectionBindings &projectionBindings) {
  // The seal keeps a bare equality immutable under this rewrite; the clone
  // rule below is the one mover, and it reaches an equality only through the
  // claim that wraps it.
  AttrTypeReplacer replacer = makeEndpointSealedReplacer();
  bool isTemplate = kind == CloneKind::Template;
  // The replacer outlives its caller's maps, so it holds its own copies, and
  // every type the clone visits reads both.
  auto variables = std::make_shared<SpecializationMap>(variableBindings);
  auto projections = std::make_shared<ProjectionBindings>(projectionBindings);
  auto stamp = [=](Type t) -> Type {
    auto others = [&](Type key) -> std::optional<Type> {
      auto projection = dyn_cast<ProjectionType>(key);
      return projection ? projections->lookup(projection) : std::nullopt;
    };
    return isTemplate
               ? applySubstitution(*variables, others, t,
                                   ClaimPredicates::VariablesAlone)
               : applySubstitutionToFixedPoint(*variables, others, t,
                                               ClaimPredicates::VariablesAlone);
  };
  replacer.addReplacement(
      [=](Type t) -> std::optional<std::pair<Type, WalkResult>> {
    // A template's clone: the substitution is a structural rewrite of the whole
    // type already, so what it stamped in is the answer and the walk does not
    // re-enter it. Re-entering is what reads a parameter's argument as though it
    // were the declaration's own spelling again, which grows a parameter bound
    // over itself one level per visit.
    if (isTemplate)
      return std::make_pair(stamp(t), WalkResult::skip());

    Type result = stamp(t);

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

  // The clone rule for a claim: its predicate receives the variable bindings
  // alone, stamped once -- no projection binding resolved inside it -- so the
  // claim a clone holds is the template's at the instance's arguments, and the
  // evidence a template wrote for it still meets it by identity
  // (`respellClaimPredicate`). A citation's stated arguments are spelled as
  // its claims are and move by the same rule, past the seal that keeps every
  // other rewrite out of them (`makeEndpointSealedReplacer`).
  auto respell = [variables](Type t) {
    return applySubstitution(*variables, nullptr, t,
                             ClaimPredicates::Substituted);
  };
  replacer.addReplacement(
      [respell](ClaimType claim) -> std::optional<std::pair<Type, WalkResult>> {
    return respellClaimPredicate(claim, respell);
  });
  replacer.addReplacement(
      [respell](ImplArgumentsAttr stated)
          -> std::optional<std::pair<Attribute, WalkResult>> {
        SmallVector<Type> moved = llvm::map_to_vector(stated.getTypes(), respell);
        return std::make_pair(
            Attribute(ImplArgumentsAttr::get(stated.getContext(), moved)),
            WalkResult::skip());
      });

  return replacer;
}

AttrTypeReplacer makeSpellingReplacerFromSubstitution(
    const SpecializationMap &variables) {
  return makeTypeReplacerFromSubstitution(variables, CloneKind::Template);
}

/// Whether the block a builder inserts into stands inside a trait, impl, or
/// proof, or a still-polymorphic function -- a template, whose clone carries no
/// projection binding and no evidence binding, because its spelling is resolved
/// when the template is itself cloned for a concrete instance.
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

/// When `function` is a `func.func` cut from a method, ends with `func.return`
/// every block of its body the method's `trait.return` ends, over the same
/// operands. A return nested in a deeper region ends no block of the body and is
/// left alone.
static void endWithFunctionReturns(RewriterBase &rewriter,
                                   FunctionOpInterface function) {
  if (!isa<func::FuncOp>(function))
    return;
  for (Block &block : function.getFunctionBody()) {
    if (block.empty())
      continue;
    auto methodReturn = dyn_cast<ReturnOp>(&block.back());
    if (!methodReturn)
      continue;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(methodReturn);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(methodReturn,
                                                methodReturn.getOperands());
  }
}

FunctionOpInterface specializePolymorph(RewriterBase &rewriter,
                                        FunctionOpInterface polymorph,
                                        StringRef instanceName,
                                        const SpecializationMap &variables,
                                        const ProjectionBindings &projections) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot specialize external function");
    return nullptr;
  }

  Location loc = polymorph.getLoc();
  OpBuilder &builder = rewriter;

  // A clone whose signature still spells a type variable under the generic-keyed
  // bindings alone, or that is inserted inside a trait, impl, or proof, is a
  // template: it is stamped under those bindings with no projection or evidence
  // binding, so a substitution-invariant verifier accepts it as it accepts the
  // source, and its spelling resolves when it is cloned for a concrete
  // instance. A monomorphic clone receives the full call substitution.
  AttrTypeReplacer variableReplacer =
      makeSpellingReplacerFromSubstitution(variables);

  auto oldFunctionType = cast<FunctionType>(polymorph.getFunctionType());
  auto substitutedType =
      llvm::cast<FunctionType>(variableReplacer.replace(oldFunctionType));

  bool cloneIsTemplate = isPolymorphicType(Type(substitutedType)) ||
                         insertionStandsInsideTemplate(builder);
  AttrTypeReplacer fullReplacer = makeTypeReplacerFromSubstitution(
      variables, CloneKind::Instance, projections);
  AttrTypeReplacer &replacer = cloneIsTemplate ? variableReplacer : fullReplacer;

  auto newFunctionType =
      cloneIsTemplate
          ? substitutedType
          : llvm::cast<FunctionType>(replacer.replace(oldFunctionType));

  // create the instance with the new type and instance name, of the kind the
  // block it stands in holds
  bool instanceIsMethod =
      isa<TraitOp, ImplOp>(builder.getInsertionBlock()->getParentOp());
  FunctionOpInterface instance =
      instanceIsMethod
          ? FunctionOpInterface(
                MethodOp::create(builder, loc, instanceName, newFunctionType))
          : FunctionOpInterface(func::FuncOp::create(builder, loc, instanceName,
                                                     newFunctionType));

  // clone the polymorph's attributes with type replacement
  for (NamedAttribute attr : polymorph->getAttrs()) {
    StringRef n = attr.getName();

    // don't copy the polymorph's name or function type, and give a method no
    // visibility
    if (n == SymbolTable::getSymbolAttrName() || n == "function_type" ||
        (instanceIsMethod && n == SymbolTable::getVisibilityAttrName())) {
      continue;
    }

    instance->setAttr(attr.getName(), replacer.replace(attr.getValue()));
  }

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph.getFunctionBody(),
                                 instance.getFunctionBody(),
                                 mapping,
                                 replacer,
                                 variableReplacer);
  endWithFunctionReturns(rewriter, instance);

  return instance;
}

void specializePolymorphicRegion(OpBuilder &builder, Region &polymorph,
                                 Region &monomorph,
                                 const SpecializationMap &subst) {
  assert(monomorph.empty() && "Region is not empty");

  // A region cloned into a template keeps its spelling: its projections
  // resolve when the template is cloned for a concrete instance, not here.
  CloneKind kind = insertionStandsInsideTemplate(builder)
                       ? CloneKind::Template
                       : CloneKind::Instance;
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst, kind);
  AttrTypeReplacer spellingReplacer = makeSpellingReplacerFromSubstitution(subst);

  IRMapping mapping;
  cloneRegionWithTypeReplacement(builder,
                                 polymorph,
                                 monomorph,
                                 mapping,
                                 replacer,
                                 spellingReplacer);
}

FailureOr<InstanceKey>
InstanceKey::get(SymbolRefAttr templateRef, ArrayRef<Type> typeArguments,
                 TypeRange formalInputs, TypeRange actualInputs,
                 AttrTypeReplacer &stamp,
                 llvm::function_ref<FailureOr<ClaimType>(ClaimType, ClaimType)> respell) {
  if (formalInputs.size() != actualInputs.size())
    return failure();

  SmallVector<Type> stampedArguments;
  for (Type argument : typeArguments)
    stampedArguments.push_back(stamp.replace(argument));

  SmallVector<Type> evidence;
  for (auto [formal, supplied] : llvm::zip(formalInputs, actualInputs)) {
    // Whether a position takes evidence is read off its formal as the instance
    // spells it: a formal spelled as a projection that resolves to a claim
    // takes evidence exactly as one spelled as that claim does.
    Type parameter = stamp.replace(formal);
    if (!containsType<ClaimType>(parameter)) {
      evidence.push_back(Type());
      continue;
    }
    Type actual = stamp.replace(supplied);
    // A proven application claim supplied under another spelling than the
    // parameter's is carried there by its proof respelled.
    auto declared = dyn_cast<ClaimType>(parameter);
    auto given = dyn_cast<ClaimType>(actual);
    if (declared && given && declared.isApplication() &&
        !declared.isProven() && given.isApplication() && given.isProven() &&
        given.asUnproven() != declared) {
      FailureOr<ClaimType> respelled = respell(given, declared);
      if (failed(respelled))
        return failure();
      evidence.push_back(*respelled);
      continue;
    }
    // A claim's predicate is a proposition, never evidence, so the walk
    // judges each claim and not what stands inside it (`walkObligationSites`).
    bool everyApplicationProven = true;
    walkObligationSites(actual, [&](Type sub) {
      if (isa<ClaimType>(sub) && isUndischargedObligation(sub))
        everyApplicationProven = false;
    });
    if (!everyApplicationProven)
      return failure();
    evidence.push_back(actual);
  }
  return InstanceKey(templateRef, stampedArguments, std::move(evidence));
}

std::string InstanceKey::getSymbolName() const {
  // The positions that take evidence are fixed by the template, so the evidence
  // that follows the type arguments is read back to its positions unambiguously.
  SmallVector<Type> identity(typeArguments);
  llvm::append_range(identity, llvm::make_filter_range(
                                   evidence, [](Type supplied) {
                                     return static_cast<bool>(supplied);
                                   }));

  // A method's template is reached through its impl, and the hash qualifies the
  // root it is reached through, so a method's name reads its impl, the identity
  // of the instance, then the method.
  std::string name = templateRef.getRootReference().str() +
                     generateMangledNameSuffixFor(identity);
  for (FlatSymbolRefAttr nested : templateRef.getNestedReferences())
    name += "_" + nested.getValue().str();
  return name;
}

func::FuncOp getOrCutInstance(RewriterBase &rewriter, ModuleOp module,
                              const InstanceKey &key,
                              llvm::function_ref<func::FuncOp(StringRef)> cut) {
  std::string name = key.getSymbolName();
  if (auto existing = lookupSymbolFrom<func::FuncOp>(
          module, FlatSymbolRefAttr::get(module.getContext(), name)))
    return existing;

  func::FuncOp instance = cut(name);
  if (!instance)
    return nullptr;

  FunctionType signature = instance.getFunctionType();
  SmallVector<Type> inputs(signature.getInputs());
  Block &entry = instance.getBody().front();
  rewriter.modifyOpInPlace(instance, [&] {
    for (auto [position, supplied] : llvm::enumerate(key.getEvidence())) {
      if (!supplied)
        continue;
      inputs[position] = supplied;
      entry.getArgument(position).setType(supplied);
    }
    instance.setFunctionType(FunctionType::get(
        instance.getContext(), inputs, signature.getResults()));
  });

  // A projection off a proven source is replaced by the evidence the source's
  // impl returns at its index (`ProjectOp::inlineEvidence`). A value the
  // substitution spelled proven is left as spelled, and one nothing here
  // proves is the stage patterns' to prove. Repeated so that a chain is read
  // in dominance order whatever order its blocks stand in, and so that a
  // projection an inlined return computes is inlined in turn; evidence with no
  // base (`ProjectOp::readEvidence`) is never inlined, so the rounds reach
  // their bound only as a tripwire.
  for (unsigned round = 0; round < kInstantiationDepthLimit; ++round) {
    bool changed = false;
    SmallVector<ProjectOp> projections;
    instance.walk([&](ProjectOp project) {
      if (project.getSourceClaim().isProven() &&
          project.readEvidence().end == EvidenceReading::End::Base)
        projections.push_back(project);
    });
    for (ProjectOp project : projections)
      changed |= succeeded(project.inlineEvidence(rewriter));
    if (!changed)
      break;
  }

  // A claim the instance returns is the value its returns hand back: where
  // every return supplies one proven claim at a position the signature spells
  // as that claim unproven, the result carries that proof, as the values
  // feeding it do.
  SmallVector<Type> results(instance.getFunctionType().getResults());
  bool refined = false;
  for (unsigned position = 0; position < results.size(); ++position) {
    auto formal = dyn_cast<ClaimType>(results[position]);
    if (!formal || !formal.isApplication() || formal.isProven())
      continue;
    llvm::SmallSetVector<Type, 1> supplied;
    instance.walk([&](func::ReturnOp ret) {
      supplied.insert(ret.getOperand(position).getType());
    });
    auto proven = supplied.size() == 1
                      ? dyn_cast<ClaimType>(supplied.front())
                      : ClaimType();
    if (!proven || !proven.isProven() || proven.asUnproven() != formal)
      continue;
    results[position] = proven;
    refined = true;
  }
  if (refined)
    rewriter.modifyOpInPlace(instance, [&] {
      instance.setFunctionType(FunctionType::get(
          instance.getContext(), instance.getFunctionType().getInputs(),
          results));
    });
  return instance;
}

} // end mlir::trait
