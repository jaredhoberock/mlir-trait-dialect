// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include "TraitOps.hpp"
#include "Trait.hpp"
#include "TraitTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::trait {

//===----------------------------------------------------------------------===//
// ConvertToTraitPass
//===----------------------------------------------------------------------===//

LogicalResult convertToTrait(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);

  // collect patterns from participating dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateConvertToTraitPatterns(patterns);
  }

  // apply patterns
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  return success();
}

void ConvertToTraitPass::runOnOperation() {
  if (failed(convertToTrait(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertToTraitPass() {
  return std::make_unique<ConvertToTraitPass>();
}


//===----------------------------------------------------------------------===//
// VerifyMonomorphsPass
//===----------------------------------------------------------------------===//

LogicalResult verifyMonomorphs(ModuleOp module) {
  // forbid monomorphic functions from mentioning !trait.claim in their signatures
  for (auto f : module.getOps<func::FuncOp>()) {
    auto fnTy = f.getFunctionType();
    if (isMonomorphicType(fnTy)) {
      if (containsType<ClaimType>(fnTy))
        return f.emitOpError() << "free function monomorphs may not contain !trait.claim types";
    }
  }
  return success();
}

void VerifyMonomorphsPass::runOnOperation() {
  if (failed(verifyMonomorphs(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createVerifyMonomorphsPass() {
  return std::make_unique<VerifyMonomorphsPass>();
}


//===----------------------------------------------------------------------===//
// VerifyAcyclicTraitsPass
//===----------------------------------------------------------------------===//

LogicalResult verifyAcyclicTraits(ModuleOp module) {
  enum class Status : uint8_t { NotSeen = 0, InPath, Done };
  DenseMap<TraitOp, Status> status;
  SmallVector<TraitOp, 16> stack;

  std::function<LogicalResult(TraitOp)> dfs = [&](TraitOp u) -> LogicalResult {
    Status &s = status[u];
    if (s == Status::InPath) {
      // back-edge: report the cycle u ... u
      auto it = llvm::find(stack, u);
      auto diag = u.emitError("cycle in trait `where` clause: ");
      for (auto i = it; i != stack.end(); ++i)
        diag << "@" << i->getSymName() << " -> ";
      diag << "@" << u.getSymName();
      return failure();
    }

    if (s == Status::Done) return success();

    s = Status::InPath;
    stack.push_back(u);

    for (auto &app : u.getRequirements()) {
      auto v = app.getTraitOrAbort(module, "verifyAcyclicTraits");
      // A requirement like @Trait[!trait.proj<@Trait[!S], "Assoc">] is a
      // syntactic self-reference, but not a real cycle: the projection resolves
      // to a concrete type during monomorphization, breaking the edge. Skip it
      // so that traits with bounded associated types (e.g. `type Assoc: Trait`)
      // don't falsely trigger the acyclicity check.
      if (v == u && containsType<ProjectionType>(app.getTypeArgs().front()))
        continue;
      if (failed(dfs(v))) return failure();
    }

    stack.pop_back();
    s = Status::Done;
    return success();
  };

  for (TraitOp t : module.getOps<TraitOp>()) {
    if (status.lookup(t) == Status::Done) continue;
    if (failed(dfs(t))) {
      return failure();
    }
  }

  return module.verify();
}

void VerifyAcyclicTraitsPass::runOnOperation() {
  if (failed(verifyAcyclicTraits(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createVerifyAcyclicTraitsPass() {
  return std::make_unique<VerifyAcyclicTraitsPass>();
}


//===----------------------------------------------------------------------===//
// ResolveImplsPass
//===----------------------------------------------------------------------===//

namespace {

static void applySubstitutionInPlace(const EvidenceBindings& evidence, Operation* root) {
  if (evidence.empty()) return;
  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(evidence.toTypeMap());
  replacer.recursivelyReplaceElementsIn(root,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

/// Proves a claim-producing op and replaces it with a trait.witness.
///
/// The proving obligation is keyed on the result ClaimType, not the
/// producing op: allege, derive, and project results all discharge through
/// this one rule. `allegeOnly` restricts matching to trait.allege for the
/// resolve-impls phase, which runs before instantiation and must not yet
/// touch claims derived inside still-polymorphic bodies.
struct ProveClaimResultPattern : public RewritePattern {
  // one ImplResolver per module; owned by the pass, passed by ref into this pattern
  ImplResolver& resolver;
  bool allegeOnly;

  ProveClaimResultPattern(MLIRContext* ctx,
                          ImplResolver &resolver,
                          bool allegeOnly)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      resolver(resolver), allegeOnly(allegeOnly) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter& rewriter) const override {
    if (allegeOnly ? !isa<AllegeOp>(op) : !isa<AllegeOp, DeriveOp, ProjectOp>(op))
      return failure();

    // an already-proven result needs no work (in-place retyping by proof
    // propagation can prove a claim out from under its producing op)
    auto claim = cast<ClaimType>(op->getResult(0).getType());
    if (claim.isProven())
      return failure();

    // skip polymorphic claims -- they can't be resolved until after monomorphization
    if (!claim.isMonomorphic())
      return rewriter.notifyMatchFailure(op, "polymorphic claim deferred");

    auto errFn = [&] { return op->emitOpError(); };

    // build or reuse canonical evidence for this claim
    auto sym = resolver.resolveAndEnsureProofFor(claim, rewriter, errFn);
    if (failed(sym))
      return rewriter.notifyMatchFailure(op, "couldn't find proof of this claim");

    // replace the producer with a witness
    rewriter.replaceOpWithNewOp<WitnessOp>(
      op,
      *sym,
      claim.getTraitApplication()
    );

    return success();
  }
};

} // end namespace

FailureOr<ImplResolver> resolveImpls(ModuleOp module) {
  // run convert-to-trait patterns
  if (failed(convertToTrait(module)))
    return failure();

  // verify that monomorphs are legal
  if (failed(verifyMonomorphs(module)))
    return failure();

  // verify traits are acyclic
  if (failed(verifyAcyclicTraits(module)))
    return failure();

  // an ImplResolver for this module
  ImplResolver resolver(module);

  MLIRContext *ctx = module.getContext();

  // apply rewrite patterns
  {
    RewritePatternSet patterns(ctx);
    patterns.add<ProveClaimResultPattern>(ctx, resolver, /*allegeOnly=*/true);

    // rewrite trait.allege -> trait.witness
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return failure();
  }

  // assert that no monomorphic trait.allege remain
  bool hasLeftovers = false;
  module.walk([&](AllegeOp op) {
    if (!op.getClaim().isMonomorphic()) return;
    hasLeftovers = true;
    op.emitError() << "unresolved monomorphic trait.allege after resolve-impls";
  });
  if (hasLeftovers) return failure();

  // Normalize claim types: after allege→witness, a proof's type parameter
  // may itself contain a claim that was just proven.  Substitute all
  // unproven claims with their proven forms so that downstream instantiation
  // sees consistent types.
  applySubstitutionInPlace(resolver.buildClaimSubstitutionFromMemo(), module);

  return resolver;
}

void ResolveImplsPass::runOnOperation() {
  if (failed(resolveImpls(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createResolveImplsPass() {
  return std::make_unique<ResolveImplsPass>();
}


//===----------------------------------------------------------------------===//
// InstantiateMonomorphsPass
//===----------------------------------------------------------------------===//

/// Extend this substitution with bindings that resolve concrete `!trait.proj`
/// types visible after applying the current substitution.
void CallSubstitution::discoverProjectionBindings(TypeRange types,
                                                  ImplResolver &resolver,
                                                  PatternRewriter &rewriter) {
  for (Type ty : types) {
    apply(ty).walk([&](Type t) {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj))
        return;
      if (projectionBindings.lookup(proj))
        return;
      if (auto resolved = resolver.resolveProjectionType(proj, rewriter);
          succeeded(resolved))
        projectionBindings.bind(proj, *resolved);
    });
  }
}

/// Record proven-claim bindings visible after applying the current
/// substitution.
LogicalResult CallSubstitution::discoverEvidenceBindings(
    TypeRange types, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  for (Type ty : types) {
    Type rewritten = apply(ty);
    if (failed(recordProofBindingsIn(rewritten, module, evidenceBindings, err)))
      return failure();
  }
  return success();
}

/// Close this substitution under projection and proof bindings.
///
/// The initial call substitution contains direct polymorphic-type bindings and
/// proof spellings visible at the call site. Projection bindings can rewrite
/// those spellings, which can reveal new proof bindings; newly recorded proof
/// bindings may in turn expose projections in their normalized type. Iterate
/// until no new component bindings are discovered so call lowering does not
/// depend on a particular phase order.
///
/// The fixed-point loop relies on disjoint component key kinds and monotone
/// binding growth. If closing fails, discard this substitution; partial
/// evidence bindings may have been recorded before the failing obligation.
LogicalResult CallSubstitution::close(
    TypeRange operandTypes, TypeRange resultTypes, FunctionType formalTy,
    ModuleOp module, ImplResolver &resolver, PatternRewriter &rewriter,
    llvm::function_ref<InFlightDiagnostic()> err) {
  bool changed;
  do {
    // The component maps grow monotonically; `bindingCount()` is the raw component sum
    // so it is not affected by fixed-point normalization of the merged map.
    size_t before = bindingCount();

    discoverProjectionBindings(resultTypes, resolver, rewriter);
    discoverProjectionBindings(operandTypes, resolver, rewriter);
    if (formalTy) {
      discoverProjectionBindings(formalTy.getInputs(), resolver, rewriter);
      discoverProjectionBindings(formalTy.getResults(), resolver, rewriter);
    }

    if (failed(discoverEvidenceBindings(operandTypes, module, err)))
      return failure();
    if (failed(discoverEvidenceBindings(resultTypes, module, err)))
      return failure();

    changed = bindingCount() != before;
  } while (changed);

  return success();
}

namespace {

/// The common product of lowering either kind of trait call site: the callee
/// specialized for this call and the result types after applying the same
/// closed substitution.
struct SpecializedCallTarget {
  func::FuncOp callee;
  SmallVector<Type> resultTypes;
};

/// Checks the operand precondition shared by trait function and method calls.
static LogicalResult requireMonomorphicOperands(Operation *op,
                                                ValueRange operands,
                                                PatternRewriter &rewriter) {
  for (Value operand : operands)
    if (isPolymorphicType(operand.getType()))
      return rewriter.notifyMatchFailure(op, "operands are still polymorphic");
  return success();
}

/// Builds and closes the call-site substitution, uses it to specialize the
/// callee, and computes the concrete result types for the replacement call.
template <typename CallOpT, typename GetFormalTy>
static FailureOr<SpecializedCallTarget>
specializeCallTarget(CallOpT op, PatternRewriter &rewriter,
                     ImplResolver &resolver, GetFormalTy getFormalTy,
                     StringRef formalTypeFailure) {
  auto subst = op.buildParameterSpecialization();
  if (failed(subst)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't build substitution");
    return failure();
  }

  auto formalTy = getFormalTy(op);
  if (failed(formalTy)) {
    (void)rewriter.notifyMatchFailure(op, formalTypeFailure);
    return failure();
  }

  ModuleOp module = op.getOperation()->template getParentOfType<ModuleOp>();
  auto errFn = [&] { return op.emitOpError(); };
  if (failed(subst->close(op.getOperandTypes(), op.getResultTypes(), *formalTy,
                          module, resolver, rewriter, errFn)))
    return failure();

  SpecializedCallTarget target;
  for (Type r : op.getResultTypes()) {
    Type newR = subst->apply(r);
    if (isPolymorphicType(newR)) {
      (void)rewriter.notifyMatchFailure(op, "result type is still polymorphic");
      return failure();
    }
    target.resultTypes.push_back(newR);
  }

  auto callee = op.getOrSpecializeCallee(rewriter, *subst);
  if (failed(callee)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't get or specialize callee");
    return failure();
  }
  target.callee = *callee;
  return target;
}

struct FuncCallOpLowering : public OpRewritePattern<FuncCallOp> {
  ImplResolver &resolver;

  FuncCallOpLowering(MLIRContext *ctx, ImplResolver &resolver)
    : OpRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(FuncCallOp callOp, PatternRewriter &rewriter) const override {
    if (failed(requireMonomorphicOperands(callOp, callOp.getOperands(), rewriter)))
      return failure();

    // func.call requires the call and callee to be in the same scope;
    // specialized callees are emitted at module scope, so only lower calls
    // already in the module's symbol table.
    Operation *nearestTable = SymbolTable::getNearestSymbolTable(callOp);
    if (!nearestTable || !isa<ModuleOp>(nearestTable))
      return rewriter.notifyMatchFailure(callOp, "call is still nested in a method");

    auto target = specializeCallTarget(
        callOp, rewriter, resolver,
        [](FuncCallOp op) { return op.getCalleeFunctionType(); },
        "couldn't get callee function type");
    if (failed(target))
      return failure();

    // Operands pass through untouched (as in MethodCallOpLowering).
    // trait.proj.cast operands are not this lowering's concern: projection
    // resolution turns them into identity casts that fold away, and
    // claim-typed survivors are erased with the claims, all within the
    // same rewrite fixpoint; verification runs after the fixpoint settles.
    rewriter.replaceOpWithNewOp<func::CallOp>(
      callOp,
      target->callee.getSymName(),
      target->resultTypes,
      callOp.getOperands()
    );

    return success();
  }
};

struct MethodCallOpLowering : public OpRewritePattern<MethodCallOp> {
  ImplResolver &resolver;

  MethodCallOpLowering(MLIRContext *ctx, ImplResolver &resolver)
    : OpRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(MethodCallOp op, PatternRewriter &rewriter) const override {
    if (failed(requireMonomorphicOperands(op, op.getOperands(), rewriter)))
      return failure();
    if (!op.getClaimType().isProven())
      return rewriter.notifyMatchFailure(op, "claim is still unproven");

    auto target = specializeCallTarget(
        op, rewriter, resolver,
        [](MethodCallOp op) { return op.getMethodFunctionType(); },
        "couldn't get method function type");
    if (failed(target))
      return failure();

    // pass the claim as the first argument to the specialized callee
    SmallVector<Value> args;
    args.push_back(op.getClaim());
    llvm::append_range(args, op.getArguments());

    // replace with a trait.func.call to the specialized callee
    rewriter.replaceOpWithNewOp<FuncCallOp>(
      op,
      target->resultTypes,
      target->callee.getSymName(),
      args
    );

    return success();
  }
};

/// Monomorphizes result types for any op implementing
/// InferTypeOpInterface once all operands are monomorphic.
///
/// When all operands have concrete (non-polymorphic) types, the op's
/// `inferReturnTypes` computes the specialized result types. If they
/// differ from the op's current result types after normalization, the
/// pattern updates them in-place under the rewriter.
struct MonomorphizeResultTypesPattern
    : public OpInterfaceRewritePattern<InferTypeOpInterface> {
  ImplResolver &resolver;

  MonomorphizeResultTypesPattern(MLIRContext *ctx, ImplResolver &resolver)
    : OpInterfaceRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(InferTypeOpInterface iface,
                                PatternRewriter &rewriter) const override {
    // InferTypeOpInterface is implemented by ops well outside this
    // dialect's orbit (arith and friends), so participation is gated on
    // having something to refine: at least one current result type is
    // non-ground (mentions a poly var, inference var, projection, or
    // claim).
    if (llvm::all_of(iface->getResultTypes(), isGroundType))
      return rewriter.notifyMatchFailure(iface, "result types are already ground");

    // only run when all operands are monomorphic
    for (Type ty : iface->getOperandTypes()) {
      if (isPolymorphicType(ty))
        return rewriter.notifyMatchFailure(iface, "operands are still polymorphic");
    }

    // try to compute specialized result types; inference failure defers this op
    SmallVector<Type> inferred;
    if (failed(iface.inferReturnTypes(iface->getContext(), iface->getLoc(),
                                      iface->getOperands(),
                                      iface->getAttrDictionary(),
                                      iface->getPropertiesStorage(),
                                      iface->getRegions(), inferred)))
      return rewriter.notifyMatchFailure(iface, "cannot infer result types from operands");
    FailureOr<SmallVector<Type>> specializedTypes = std::move(inferred);

    // the arity of results must match
    if (specializedTypes->size() != iface->getNumResults())
      return rewriter.notifyMatchFailure(iface, "specialized result type count mismatch");

    // The interface computes result types from the operands' CURRENT
    // spellings and promises nothing about their normal form, while sibling
    // patterns rewrite spellings in place (projection -> resolved,
    // unproven -> proven claim). Writing an operand-echoed spelling over an
    // already-normalized result would undo those patterns and livelock the
    // greedy driver, so every computed type is normalized before the
    // changed-check: this pattern may move result types toward normal form,
    // never away from it. Trait infrastructure regions are templates whose
    // projections legitimately stay symbolic, so they skip normalization.
    // Note resolveProjectionsIn may mint impls/proofs through the rewriter
    // even when the pattern then reports "result types unchanged"
    // (precedented by ResolveProjectionsPattern's wouldReplace probe).
    if (!iface->getParentOfType<TraitOp>() && !iface->getParentOfType<ImplOp>()) {
      AttrTypeReplacer proofs = makeTypeReplacerFromSubstitution(
          resolver.buildClaimSubstitutionFromMemo().toTypeMap());
      for (Type &ty : *specializedTypes)
        ty = proofs.replace(resolver.resolveProjectionsIn(ty, rewriter));
    }

    // check if anything actually changes
    if (llvm::equal(iface->getResultTypes(), *specializedTypes))
      return rewriter.notifyMatchFailure(iface, "result types unchanged");

    // mutate result types in-place
    rewriter.modifyOpInPlace(iface, [&] {
      for (auto [result, newType] : llvm::zip(iface->getResults(), *specializedTypes))
        result.setType(newType);
    });

    return success();
  }
};

/// Returns true if `replacer.replaceElementsIn(op, ...)` with the given
/// options would modify anything on `op` (not recursing into children).
static bool wouldReplace(AttrTypeReplacer &replacer, Operation *op,
                         bool replaceAttrs, bool replaceLocs, bool replaceTypes) {
  if (replaceTypes) {
    for (Type t : op->getResultTypes())
      if (replacer.replace(t) != t) return true;
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (Value arg : b.getArguments())
          if (replacer.replace(arg.getType()) != arg.getType()) return true;
  }
  if (replaceAttrs)
    for (NamedAttribute attr : op->getAttrs())
      if (replacer.replace(attr.getValue()) != attr.getValue()) return true;
  if (replaceLocs)
    if (replacer.replace(op->getLoc()) != op->getLoc()) return true;
  return false;
}

/// Propagates proofs from the resolver's memo into claim types of any op
/// that carries unproven claims the resolver can now prove.
///
/// When ProveClaimResultPattern proves a claim during the greedy rewrite and
/// FuncCallOpLowering subsequently instantiates a callee that expects that
/// claim, the newly created ops carry unproven claim types.  This pattern
/// substitutes those with their proven counterparts, unblocking
/// MethodCallOpLowering within the same rewrite pass.
///
/// Only replaces types owned by the matched op itself (result types, block
/// argument types, and attributes).  Operand types are SSA-determined and
/// update automatically once the defining value carries the proven type.
/// Child ops are visited independently by the greedy driver.
struct PropagateProofsPattern : public RewritePattern {
  ImplResolver &resolver;

  PropagateProofsPattern(MLIRContext *ctx, ImplResolver &resolver)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      resolver(resolver) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto evidence = resolver.buildClaimSubstitutionFromMemo();
    if (evidence.empty())
      return failure();

    if (!opMentionsType<ClaimType>(op))
      return failure();

    AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(evidence.toTypeMap());
    if (!wouldReplace(replacer, op,
                      /*replaceAttrs=*/true,
                      /*replaceLocs=*/false,
                      /*replaceTypes=*/true))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      replacer.replaceElementsIn(op,
                                 /*replaceAttrs=*/true,
                                 /*replaceLocs=*/false,
                                 /*replaceTypes=*/true);
    });
    return success();
  }
};

/// Resolves concrete `!trait.proj` types to their bound types by looking up
/// the matching `trait.impl`'s associated type binding.
struct ResolveProjectionsPattern : public RewritePattern {
  ImplResolver &resolver;

  ResolveProjectionsPattern(MLIRContext *ctx, ImplResolver &resolver)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip trait infrastructure ops and their children -
    // they may legitimately contain projections
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return failure();
    if (op->getParentOfType<TraitOp>() || op->getParentOfType<ImplOp>())
      return failure();

    if (!opMentionsType<ProjectionType>(op))
      return failure();

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](Type t) -> std::optional<Type> {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj)) return std::nullopt;
      auto resolved = resolver.resolveProjectionType(proj, rewriter);
      if (failed(resolved)) return std::nullopt;
      return *resolved;
    });
    if (!wouldReplace(replacer, op,
                      /*replaceAttrs=*/true,
                      /*replaceLocs=*/false,
                      /*replaceTypes=*/true))
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      replacer.replaceElementsIn(op,
                                 /*replaceAttrs=*/true,
                                 /*replaceLocs=*/false,
                                 /*replaceTypes=*/true);
    });
    return success();
  }
};

} // end namespace

LogicalResult instantiateMonomorphs(ModuleOp module) {
  // resolve impls first
  auto resolver = resolveImpls(module);
  if (failed(resolver))
    return failure();

  MLIRContext* ctx = module.getContext();

  // rewrite trait.func.call and trait.method.call, prove claim producers
  // (allege, derive, project), resolve projections, and monomorphize any
  // generic op whose results become monomorphic
  RewritePatternSet patterns(ctx);
  patterns.add<ProveClaimResultPattern>(ctx, *resolver, /*allegeOnly=*/false);
  patterns.add<MonomorphizeResultTypesPattern>(ctx, *resolver);
  patterns.add<FuncCallOpLowering>(ctx, *resolver);
  patterns.add<MethodCallOpLowering>(ctx, *resolver);
  patterns.add<ResolveProjectionsPattern>(ctx, *resolver);
  patterns.add<PropagateProofsPattern>(ctx, *resolver);

  // collect instantiate-monomorphs patterns from other dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateInstantiateMonomorphsPatterns(patterns);
  }

  // Bound the total rewrite count so that a non-confluent pattern pair
  // fails loudly instead of livelocking. The driver's iteration limit
  // cannot catch a livelock: two patterns that keep undoing each other's
  // in-place type rewrites hold the worklist non-empty WITHIN one
  // iteration. The bound scales with input size; legitimate runs rewrite
  // each op a small bounded number of times as types refine, so any run
  // that reaches the bound is cycling.
  int64_t opCount = 0;
  module.walk([&](Operation *) { ++opCount; });
  GreedyRewriteConfig config;
  config.setMaxNumRewrites(opCount * 1024 + 4096);

  // apply patterns
  if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
    return module.emitError(
        "instantiate-monomorphs did not converge: rewrite budget exceeded, "
        "which indicates a non-confluent pattern pair cycling on a type "
        "spelling");

  // Assert that no op produced an unproven monomorphic claim that escaped
  // proving. Keying this check on the result ClaimType rather than on the
  // set of claim-producing ops makes it total over producers: an op whose
  // claims the patterns above fail to discharge is an error here, never a
  // silent gap. Trait infrastructure regions are templates and keep their
  // unproven claims.
  bool hasLeftovers = false;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return WalkResult::skip();
    for (Value result : op->getResults()) {
      auto claim = dyn_cast<ClaimType>(result.getType());
      if (!claim || claim.isProven() || !claim.isMonomorphic())
        continue;
      hasLeftovers = true;
      op->emitError() << "unproven monomorphic claim " << claim
                      << " after instantiate-monomorphs";
    }
    return WalkResult::advance();
  });
  if (hasLeftovers) return failure();

  return module.verify();
}

void InstantiateMonomorphsPass::runOnOperation() {
  if (failed(instantiateMonomorphs(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createInstantiateMonomorphsPass() {
  return std::make_unique<InstantiateMonomorphsPass>();
}


//===----------------------------------------------------------------------===//
// MonomorphizePass
//===----------------------------------------------------------------------===//

namespace {

struct EraseWitnessOp : public OpRewritePattern<WitnessOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WitnessOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseProjectOp : public OpRewritePattern<ProjectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ProjectOp op, PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct EraseProjCastOp : public OpConversionPattern<ProjCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ProjCastOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // After projection resolution, input and result have the same
    // concrete type. If the input survived conversion (regular value),
    // forward it. If the input was erased (claim type mapped to 0
    // values, or defining op erased by another pattern), erase the
    // proj_cast too — there is nothing to forward.
    auto input = adaptor.getInput();
    if (input.empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, input);
    }
    return success();
  }
};


/// Erases all residual polymorphism from the module.
///
/// This runs in two phases because no single MLIR mechanism can handle
/// both kinds of work:
///
/// Phase 1 (applyPartialConversion): Structural op rewrites that erase
///   SSA values.  Claim types map to zero results (1:0 erasure), so ops
///   that carry claims need their operand lists, indices, and signatures
///   rewritten.  Only applyPartialConversion can do this — it manages
///   the value-level bookkeeping (dropping operands, remapping uses).
///   The tuple dialect adjusts tuple.get indices and tuple.make operands;
///   the func dialect rewrites function signatures and call sites.
///
/// Phase 2 (recursivelyReplaceElementsIn): Bulk type rewriting.
///   applyPartialConversion only touches operand/result types on ops
///   matched by patterns.  Types inside attributes (e.g. the body
///   TypeAttr on nominal.def) are invisible to it.  This sweep rewrites
///   all remaining types everywhere.  The nominal dialect registers its
///   NominalType name mangling here.
///
/// Each dialect contributes to both phases via populateErasePolymorphsPatterns.
static LogicalResult erasePolymorphs(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  // Delete trait symbol infrastructure upfront — these are templates that
  // have already been instantiated, and their regions contain polymorphic
  // types that would trip the legality check if left for
  // applyPartialConversion.
  for (Operation &op : llvm::make_early_inc_range(*module.getBody())) {
    if (isa<ProofOp, ImplOp, TraitOp>(op))
      op.erase();
    else if (auto f = dyn_cast<func::FuncOp>(op))
      if (isPolymorphicType(f.getFunctionType()))
        f.erase();
  }

  // Phase 1: structural op rewrites via applyPartialConversion.
  // ClaimType maps to zero results (the SSA value disappears).
  TypeConverter opConverter;
  opConverter.addConversion([](Type ty) { return ty; });
  opConverter.addConversion([](ClaimType ty, SmallVectorImpl<Type> &out) {
    return success();
  });

  AttrTypeReplacer typeSweep;

  // Collect from participating dialects
  RewritePatternSet patterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateErasePolymorphsPatterns(opConverter, patterns, typeSweep);
  }

  // Add trait dialect's own patterns
  patterns.add<EraseProjectOp, EraseWitnessOp>(ctx);
  patterns.add<EraseProjCastOp>(opConverter, ctx);

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, opConverter);
  populateCallOpTypeConversionPattern(patterns, opConverter);
  populateReturnOpTypeConversionPattern(patterns, opConverter);

  // Mark !trait.claim and !trait.proj as illegal
  ConversionTarget target(*ctx);
  target.addIllegalOp<AllegeOp, DeriveOp, ProjectOp, WitnessOp, ProjCastOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return !opMentionsType<ClaimType>(op) && !opMentionsType<ProjectionType>(op);
  });

  // Apply Phase 2
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return failure();

  // Phase 2: bulk type rewriting via recursivelyReplaceElementsIn.
  // The typeSweep replacer was already populated by dialects above
  // (e.g. nominal registered NominalType mangling).  Also forward
  // the opConverter's conversions so ClaimType gets swept out of
  // attributes too.
  typeSweep.addReplacement([&](Type t) -> std::optional<Type> {
    Type converted = opConverter.convertType(t);
    if (!converted || converted == t)
      return std::nullopt;
    return converted;
  });
  typeSweep.recursivelyReplaceElementsIn(module,
                                         /*replaceAttrs=*/true,
                                         /*replaceLocs=*/false,
                                         /*replaceTypes=*/true);
  return module.verify();
}

}

LogicalResult monomorphize(ModuleOp module) {
  if (failed(instantiateMonomorphs(module)))
    return failure();

  return erasePolymorphs(module);
}

void MonomorphizePass::runOnOperation() {
  if (failed(monomorphize(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createMonomorphizePass() {
  return std::make_unique<MonomorphizePass>();
}


} // end mlir::trait
