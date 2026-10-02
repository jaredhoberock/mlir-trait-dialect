// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "SymbolLookup.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>

namespace mlir::trait {

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
  // A replacer stamps one clone, which adds functions and no impl, so the impls
  // every lookup below scans are the same for all of them: each application's
  // candidates are read once per replacer.
  auto candidates = std::make_shared<ImplCandidateMemo>();
  // One type stamped whole: the substitution, then -- for a monomorphic clone --
  // the ground projections it minted resolved.
  auto stamp = [=](Type t) -> Type {
    if (!module)
      return applySubstitutionOnce(subst, t);
    return resolveProjectionsByLookup(applySubstitutionToFixedPoint(subst, t),
                                      module, DemandOrigin::MonomorphStampOut,
                                      LookupScope::Ground, *candidates);
  };
  replacer.addReplacement(
      [=](Type t) -> std::optional<std::pair<Type, WalkResult>> {
    // A template's clone: the substitution is a structural rewrite of the whole
    // type already, so what it stamped in is the answer and the walk does not
    // re-enter it. Re-entering is what reads a parameter's argument as though it
    // were the declaration's own spelling again, which grows a parameter bound
    // over itself one level per visit.
    if (!module)
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

  // A binding's key is a parameter of the declaration a citation names -- the
  // impl a derive or proof states arguments for -- and not of the declaration
  // this clone is cut from, whatever label the two share. So the clone stamps
  // the argument whole and keeps the key.
  replacer.addReplacement(
      [stamp](TypeBindingAttr binding)
          -> std::optional<std::pair<Attribute, WalkResult>> {
    auto stamped = TypeBindingAttr::get(binding.getContext(),
                                        binding.getParameter(),
                                        stamp(binding.getArgument()));
    return std::make_pair(Attribute(stamped), WalkResult::skip());
  });

  // The clone rule for equality evidence: an equality claim's endpoints, and a
  // projection-resolution witness's endpoints and arguments, receive the
  // variable bindings alone, stamped once -- no projection or evidence binding,
  // and no module lookup, resolved inside them -- so the witness a clone holds
  // is rebuilt at the instance its claim is, under the one substitution.
  llvm::DenseMap<Type, Type> variableBindings;
  for (auto [key, value] : subst)
    if (isa<GenericTypeInterface>(key))
      variableBindings.try_emplace(key, value);
  auto respell = [variableBindings](Type t) {
    return applySubstitutionOnce(variableBindings, t);
  };
  replacer.addReplacement(
      [respell](ClaimType claim) -> std::optional<std::pair<Type, WalkResult>> {
    return respellEqualityEndpoints(claim, respell);
  });
  replacer.addReplacement(
      [respell](WitnessAttr witness)
          -> std::optional<std::pair<Attribute, WalkResult>> {
    return respellWitness(witness, respell);
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

FunctionOpInterface specializePolymorph(RewriterBase& rewriter,
                                        FunctionOpInterface polymorph,
                                        StringRef instanceName,
                                        const DenseMap<Type,Type> &substitution) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot specialize external function");
    return nullptr;
  }

  Location loc = polymorph.getLoc();
  OpBuilder &builder = rewriter;

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

  auto oldFunctionType = cast<FunctionType>(polymorph.getFunctionType());
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
                                 replacer);
  endWithFunctionReturns(rewriter, instance);

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

FailureOr<InstanceKey> InstanceKey::get(SymbolRefAttr templateRef,
                                        ArrayRef<Type> typeArguments,
                                        TypeRange formalInputs,
                                        TypeRange actualInputs,
                                        AttrTypeReplacer &stamp) {
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
    if (!containsType<ClaimType>(stamp.replace(formal))) {
      evidence.push_back(Type());
      continue;
    }
    Type actual = stamp.replace(supplied);
    // An equality's endpoints are a proposition, never evidence, so the walk
    // judges the equality claim and not what stands inside it.
    bool everyApplicationProven = true;
    actual.walk<WalkOrder::PreOrder>([&](Type sub) -> WalkResult {
      auto claim = dyn_cast<ClaimType>(sub);
      if (!claim)
        return WalkResult::advance();
      if (claim.isEquality())
        return WalkResult::skip();
      if (!claim.isProven())
        everyApplicationProven = false;
      return WalkResult::advance();
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
                              llvm::function_ref<func::FuncOp(StringRef)> cut,
                              const EvidenceBindings &evidence) {
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

  // A value an op derives from an operand carries that operand's evidence. The
  // substitution the instance was stamped under spelled each claim one way
  // wherever it stands -- the one proof the use supplies for it, or no proof
  // where the use supplies two -- so a derived value is read off the value it
  // is derived from wherever that reading is positional:
  // - a projection the substitution left unproven off a proven source is the
  //   subproof the source's proof cites at the projection's index, where proof
  //   derivation accepts it, at the application the result spells -- the
  //   trait states the requirement before its projections are resolved, and
  //   the result's spelling is the one its readers hold;
  // - a coerce's result the substitution left unproven is its input's proof at
  //   the application the result spells; the coerce verifier refuses any other.
  // A value the substitution spelled proven is left as spelled: every reader
  // of it already holds that spelling, and the instantiation driver judges it
  // against its source before it is witnessed. Iterated to a fixed point so
  // that a chain is read in dominance order whatever order its blocks stand in.
  bool changed;
  do {
    changed = false;
    instance.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto retype = [&](Value value, Type type) {
        if (value.getType() == type)
          return;
        rewriter.modifyOpInPlace(op, [&] { value.setType(type); });
        changed = true;
      };
      if (auto project = dyn_cast<ProjectOp>(op)) {
        ClaimType result = project.getResultClaim();
        ClaimType source = project.getSourceClaim();
        if (!source.isProven() || !result.isApplication())
          return;
        if (result.isProven())
          return;
        auto requirement = getClaimRequirementAt(
            source, module, project.getIndex(), project.getBinderArguments());
        if (failed(requirement) || !requirement->conclusion.isProven())
          return;
        EvidenceBindings accepted;
        if (failed(verifyAndRecordProof(result, requirement->conclusion, module,
                                        accepted, DemandOrigin::ProofRecording,
                                        /*memo=*/nullptr, /*err=*/nullptr)) ||
            accepted.bindingCount() == 0)
          return;
        retype(project.getResult(),
               ClaimType::get(op->getContext(), result.getTraitApplication(),
                              requirement->conclusion.getProof()));
        return;
      }
      if (auto coerce = dyn_cast<CoerceOp>(op)) {
        auto input = dyn_cast<ClaimType>(coerce.getInput().getType());
        auto result = dyn_cast<ClaimType>(coerce.getResult().getType());
        if (input && result && input.isApplication() && input.isProven() &&
            result.isApplication() && !result.isProven())
          retype(coerce.getResult(),
                 ClaimType::get(op->getContext(), result.getTraitApplication(),
                                input.getProof()));
      }
    });
  } while (changed);

  // A value still spelled unproven whose claim the use supplies two proofs of.
  // The substitution could not say which, no rule above supplied one from a
  // position, and selection, asked for it later, would not know which position
  // the value stands for either: it refuses where the two proofs select two
  // impls and otherwise answers with a proof neither position supplied. The
  // instance is refused here, naming both proofs, and the call that wanted it
  // is left standing.
  //
  // XXX TODO: this refusal and the reading of derived values above are deleted
  // with the respelling of a cut body's claims by their spelling, when
  // declarations and proofs take their evidence as block arguments and a cut
  // maps each claim value from the value that supplies it.
  auto disputedClaimIn = [&](Type type) -> ClaimType {
    ClaimType found;
    type.walk([&](ClaimType claim) {
      if (!found && claim.isApplication() && !claim.isProven() &&
          evidence.proofsOf(claim).size() > 1)
        found = claim;
    });
    return found;
  };
  auto report = [&](Operation *at, Value value, ClaimType claim) {
    InFlightDiagnostic diagnostic =
        at->emitOpError() << "is left with " << value.getType()
                          << ", and this instance is supplied " << claim
                          << " by two proofs, ";
    llvm::interleave(
        evidence.proofsOf(claim), diagnostic,
        [&](ClaimType proof) { diagnostic << proof.getProof(); }, " and ");
    diagnostic << "; no position says which this value carries";
  };
  WalkResult refused = instance.walk([&](Operation *op) -> WalkResult {
    for (Value result : op->getResults())
      if (ClaimType claim = disputedClaimIn(result.getType())) {
        report(op, result, claim);
        return WalkResult::interrupt();
      }
    for (Region &region : op->getRegions())
      for (Block &block : region) {
        // The entry block's arguments are the parameters, which take the
        // evidence the key holds at their positions above.
        if (op == instance.getOperation() && &block == &region.front())
          continue;
        for (BlockArgument argument : block.getArguments())
          if (ClaimType claim = disputedClaimIn(argument.getType())) {
            report(op, argument, claim);
            return WalkResult::interrupt();
          }
      }
    return WalkResult::advance();
  });
  if (refused.wasInterrupted()) {
    rewriter.eraseOp(instance);
    return nullptr;
  }
  return instance;
}

} // end mlir::trait
