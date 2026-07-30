// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include "TraitOps.hpp"
#include "Trait.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/ScopeExit.h>
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
// convertToTrait
//===----------------------------------------------------------------------===//

namespace {

/// Counts the rewrite events one run of a greedy pattern driver raises.
///
/// The driver notifies the listener its configuration names of every op it
/// inserts, modifies in place, replaces and erases, and of every pattern
/// application that succeeded. The applications are what the driver's rewrite
/// budget bounds, so reporting them beside the budget says both how much a run
/// rewrote and how much room it had left.
struct RewriteEventCounts : public RewriterBase::Listener {
  using RewriterBase::Listener::notifyOperationReplaced;

  void notifyOperationInserted(Operation *, OpBuilder::InsertPoint) override {
    ++inserted;
  }
  void notifyOperationModified(Operation *) override { ++modified; }
  void notifyOperationReplaced(Operation *, ValueRange) override { ++replaced; }
  void notifyOperationErased(Operation *) override { ++erased; }
  void notifyPatternEnd(const Pattern &, LogicalResult status) override {
    if (succeeded(status))
      ++applications;
  }

  /// Writes this run's line, naming the driver that raised the events and the
  /// budget it ran under. A driver with no budget reports none and no headroom
  /// either, rather than a number derived from the sentinel that stands for
  /// "unbounded".
  void report(StringRef driver, int64_t budget) const {
    llvm::errs() << stageRecordRewritesPrefix << " driver=" << driver
                 << " inserted=" << inserted << " modified=" << modified
                 << " replaced=" << replaced << " erased=" << erased
                 << " applications=" << applications;
    if (budget < 0)
      llvm::errs() << " budget=unbounded headroom=unbounded\n";
    else
      llvm::errs() << " budget=" << budget
                   << " headroom=" << budget - int64_t(applications) << "\n";
  }

  uint64_t inserted = 0;
  uint64_t modified = 0;
  uint64_t replaced = 0;
  uint64_t erased = 0;
  uint64_t applications = 0;
};

/// Runs one greedy pattern driver over `root`, reporting the rewrite events it
/// raised under the name `driver`.
///
/// No listener is installed unless the counts will be reported, so a run that
/// nobody is counting costs exactly what it did before.
LogicalResult applyPatternsGreedilyAndReport(Operation *root,
                                             RewritePatternSet &&patterns,
                                             GreedyRewriteConfig config,
                                             StringRef driver) {
  RewriteEventCounts events;
  bool reporting = DemandLedger::isRecordingEnabled();
  if (reporting)
    config.setListener(&events);

  LogicalResult result = applyPatternsGreedily(root, std::move(patterns), config);

  if (reporting)
    events.report(driver, config.getMaxNumRewrites());
  return result;
}

} // namespace

LogicalResult convertToTrait(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);

  // collect patterns from participating dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateConvertToTraitPatterns(patterns);
  }

  // apply patterns
  if (failed(applyPatternsGreedilyAndReport(module, std::move(patterns),
                                            GreedyRewriteConfig(),
                                            "convert-to-trait")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// verifyMonomorphs
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

/// Verify that every proven claim spelled in a top-level function signature is
/// proven by the proof it names. A `by @proof` in a declared type is otherwise
/// checked nowhere until a call reaches it, so a signature can name a proof that
/// does not specialize to its claim and go undiagnosed. Only module-level
/// `func.func` signatures are walked; signatures nested inside trait/impl
/// method bodies are not yet covered.
LogicalResult verifyDeclaredClaimProofs(ModuleOp module) {
  LogicalResult status = success();
  for (auto f : module.getOps<func::FuncOp>()) {
    auto errFn = [&] {
      return f.emitOpError() << "declared claim in signature has an invalid proof: ";
    };
    // The obligation recorder below normalizes through the ground-projection
    // lookup, so this check raises demand of its own. The frame gives that
    // demand the signature it came from; without one it would be recorded
    // unattributed even though this dialect knows exactly where it arose.
    DemandFrame frame(f.getLoc());
    Type(f.getFunctionType()).walk([&](Type t) {
      if (status.failed())
        return;
      auto claim = dyn_cast<ClaimType>(t);
      if (!claim || !claim.isProven())
        return;
      EvidenceBindings bindings;
      if (failed(verifyAndRecordProof(claim.asUnproven(), claim, module, bindings,
                                      DemandOrigin::ProofRecording, errFn)))
        status = failure();
    });
  }
  return status;
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

  DemandRecordingSuspension verifying;
  return module.verify();
}

void VerifyAcyclicTraitsPass::runOnOperation() {
  if (failed(verifyAcyclicTraits(getOperation())))
    signalPassFailure();
}


//===----------------------------------------------------------------------===//
// ResolveImplsPass
//===----------------------------------------------------------------------===//

namespace {

/// Respells throughout `root` every claim `evidence` has a proven spelling for,
/// and reports how much of `root` that sweep touched.
///
/// The replacer's recursive entry point is this walk, so driving the walk here
/// costs nothing extra and is what lets an op the sweep respelled be told from
/// one it left alone. A position is a result type, a block-argument type, or the
/// attribute dictionary of one op; an op counts once however many of its
/// positions moved.
static void applySubstitutionInPlace(const EvidenceBindings& evidence, Operation* root) {
  if (evidence.empty()) return;
  DenseMap<Type, Type> substitution = evidence.toTypeMap();
  AttrTypeReplacer replacer =
      makeTypeReplacerFromSubstitution(substitution, /*module=*/ModuleOp());

  bool reporting = DemandLedger::isRecordingEnabled();
  uint64_t opsRespelled = 0;
  uint64_t positionsRespelled = 0;

  root->walk([&](Operation *op) {
    SmallVector<Type, 8> before;
    DictionaryAttr attributesBefore;
    auto eachTypePosition = [&](llvm::function_ref<void(Type)> visit) {
      for (Type type : op->getResultTypes())
        visit(type);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            visit(argument.getType());
    };
    if (reporting) {
      eachTypePosition([&](Type type) { before.push_back(type); });
      attributesBefore = op->getAttrDictionary();
    }

    replacer.replaceElementsIn(op,
                               /*replaceAttrs=*/true,
                               /*replaceLocs=*/false,
                               /*replaceTypes=*/true);

    if (!reporting)
      return;
    uint64_t movedHere = op->getAttrDictionary() == attributesBefore ? 0 : 1;
    size_t position = 0;
    eachTypePosition([&](Type type) {
      if (type != before[position++])
        ++movedHere;
    });
    positionsRespelled += movedHere;
    if (movedHere)
      ++opsRespelled;
  });

  if (reporting)
    llvm::errs() << stageRecordRespellingPrefix
                 << " bindings=" << substitution.size()
                 << " ops=" << opsRespelled
                 << " positions=" << positionsRespelled << "\n";
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

    DemandFrame frame(op->getLoc());

    auto errFn = [&] { return op->emitOpError(); };

    // build or reuse canonical evidence for this claim
    auto sym = resolver.resolveAndEnsureProofFor(claim, rewriter, errFn);
    if (failed(sym))
      return rewriter.notifyMatchFailure(op, "couldn't find proof of this claim");

    // Mint the witness at the same spelling the proof was recorded under.
    // Impl selection resolves the claim's monomorphic projections before
    // recording (resolveImplFor), so the recorded fact is spelled with those
    // projections resolved. Spelling the witness at the producer's source claim
    // instead would leave the witnessed application and its recorded proof
    // disagreeing on the projections. Resolving here is deterministic recorded
    // lookup (the impls are already in the module) and is idempotent with the
    // resolution resolveAndEnsureProofFor just performed.
    auto recorded = cast<ClaimType>(resolver.resolveProjectionsIn(claim, rewriter));
    rewriter.replaceOpWithNewOp<WitnessOp>(
      op,
      *sym,
      recorded.getTraitApplication()
    );

    return success();
  }
};

} // end namespace

FailureOr<ImplResolver> resolveImpls(ModuleOp module) {
  // The ledger is installed before the first sub-phase that can raise a demand:
  // conversion runs other dialects' patterns, and the declared-proof check
  // reaches the ground projection lookup through the obligation recorder.
  auto ledger = std::make_shared<DemandLedger>();
  DemandLedgerScope recording(*ledger);

  // A failing sub-phase below discards this ledger, so what it observed is
  // written out here. On success the caller keeps the ledger and reports once
  // the whole stage is done, which is why exactly one census reaches a reader.
  bool handedOn = false;
  auto census = llvm::scope_exit([&] {
    if (handedOn)
      return;
    ledger->reportServedDrainableKeys(module);
    if (DemandLedger::isRecordingEnabled())
      ledger->dumpCensus();
  });

  // run convert-to-trait patterns
  if (failed(convertToTrait(module)))
    return failure();

  // verify that monomorphs are legal
  if (failed(verifyMonomorphs(module)))
    return failure();

  // verify traits are acyclic
  if (failed(verifyAcyclicTraits(module)))
    return failure();

  // verify that proofs named in declared signatures actually prove their claims
  if (failed(verifyDeclaredClaimProofs(module)))
    return failure();

  // an ImplResolver for this module
  ImplResolver resolver(module, ledger);

  // The facts this resolver recorded are reported wherever its census is, and
  // it is the caller that reports both once the resolver is handed on. This
  // guard is declared after the resolver so that it runs while the resolver is
  // still alive, and it covers the failing exits below for the same reason the
  // census guard above covers them.
  auto facts = llvm::scope_exit([&] {
    if (handedOn || !DemandLedger::isRecordingEnabled())
      return;
    resolver.reportRecordedFacts();
  });

  MLIRContext *ctx = module.getContext();

  // apply rewrite patterns
  {
    RewritePatternSet patterns(ctx);
    patterns.add<ProveClaimResultPattern>(ctx, resolver, /*allegeOnly=*/true);

    // rewrite trait.allege -> trait.witness
    if (failed(applyPatternsGreedilyAndReport(module, std::move(patterns),
                                              GreedyRewriteConfig(),
                                              "resolve-impls")))
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

  handedOn = true;
  return resolver;
}

void ResolveImplsPass::runOnOperation() {
  auto resolver = resolveImpls(getOperation());
  if (failed(resolver)) {
    signalPassFailure();
    return;
  }

  // This pass discards its resolver, so its ledger is checked and written out
  // here. Run inside instantiate-monomorphs, the same ledger spans both
  // sub-phases and is reported once at the end of that pass instead.
  resolver->getDemandLedger().reportServedDrainableKeys(getOperation());
  if (DemandLedger::isRecordingEnabled()) {
    resolver->getDemandLedger().dumpCensus();
    resolver->reportRecordedFacts();
  }
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
                                                  OpBuilder &builder) {
  for (Type ty : types) {
    apply(ty).walk([&](Type t) {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj))
        return;
      if (projectionBindings.lookup(proj))
        return;
      // Failure adds no binding, so the call site closes over a projection it
      // still cannot spell concretely.
      auto resolved = resolver.resolveProjectionType(proj, builder);
      if (failed(resolved)) {
        recordResolverProjectionMiss(Type(proj));
        return;
      }
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
    // Closing a call substitution needs the stage's resolver, so this walk
    // runs nowhere else and its demand is the stage's.
    if (failed(recordProofBindingsIn(rewritten, module, evidenceBindings,
                                     DemandOrigin::ProofRecording, err)))
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
    ModuleOp module, ImplResolver &resolver, OpBuilder &builder,
    llvm::function_ref<InFlightDiagnostic()> err) {
  bool changed;
  do {
    // The component maps grow monotonically; `bindingCount()` is the raw component sum
    // so it is not affected by fixed-point normalization of the merged map.
    size_t before = bindingCount();

    discoverProjectionBindings(resultTypes, resolver, builder);
    discoverProjectionBindings(operandTypes, resolver, builder);
    if (formalTy) {
      discoverProjectionBindings(formalTy.getInputs(), resolver, builder);
      discoverProjectionBindings(formalTy.getResults(), resolver, builder);
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

/// Defers call lowering while any operand is a monomorphic claim that is not
/// yet proven.
///
/// Lowering a call specializes its callee against the call's argument claims,
/// and the callee body may discharge those claims through their proofs. If the
/// callee is specialized while an argument claim is still unproven, the clone
/// bakes in an unprovable parameter and any method call the body makes through
/// it cannot be resolved. An argument claim's proof can land after the call
/// first becomes eligible -- for instance a forwarded claim whose proj.cast
/// result inherits the input's proof and then folds -- so waiting for every
/// monomorphic operand claim to be proven makes callee specialization
/// independent of the order in which operand proofs settle. This mirrors the
/// existing requirement that a method call's self claim be proven before it
/// lowers. An operand claim that never becomes proven is caught downstream: the
/// leftover check keys on op results, so an unprovable claim that is an op
/// result is diagnosed there, while a claim that exists only as the block
/// argument of a still-polymorphic template is pruned by full monomorphization
/// instead.
static LogicalResult requireProvenClaimOperands(Operation *op,
                                                ValueRange operands,
                                                PatternRewriter &rewriter) {
  for (Value operand : operands)
    if (auto claim = dyn_cast<ClaimType>(operand.getType()))
      if (claim.isMonomorphic() && !claim.isProven())
        return rewriter.notifyMatchFailure(op, "operand claim is still unproven");
  return success();
}

/// Builds and closes the call-site substitution, uses it to specialize the
/// callee, and computes the concrete result types for the replacement call.
template <typename CallOpT, typename GetFormalTy>
static FailureOr<SpecializedCallTarget>
specializeCallTarget(CallOpT op, PatternRewriter &rewriter,
                     ImplResolver &resolver, GetFormalTy getFormalTy,
                     StringRef formalTypeFailure) {
  ModuleOp module = op.getOperation()->template getParentOfType<ModuleOp>();

  // Pass time: pass the module so binding a generic mid-solve resolves the
  // ground redex it mints (the module-capable comparator, not the verifier's
  // module-free one).
  auto subst = op.buildParameterSpecialization(module);
  if (failed(subst)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't build substitution");
    return failure();
  }

  auto formalTy = getFormalTy(op);
  if (failed(formalTy)) {
    (void)rewriter.notifyMatchFailure(op, formalTypeFailure);
    return failure();
  }

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
    if (failed(requireProvenClaimOperands(callOp, callOp.getOperands(), rewriter)))
      return failure();

    // func.call requires the call and callee to be in the same scope;
    // specialized callees are emitted at module scope, so only lower calls
    // already in the module's symbol table.
    Operation *nearestTable = SymbolTable::getNearestSymbolTable(callOp);
    if (!nearestTable || !isa<ModuleOp>(nearestTable))
      return rewriter.notifyMatchFailure(callOp, "call is still nested in a method");

    DemandFrame frame(callOp.getLoc());

    auto target = specializeCallTarget(
        callOp, rewriter, resolver,
        [](FuncCallOp op) { return op.getCalleeFunctionType(); },
        "couldn't get callee function type");
    if (failed(target))
      return failure();

    // Operands pass through untouched (as in MethodCallOpLowering). The
    // requireProvenClaimOperands guard above has already established that every
    // operand claim is proven, so specialization never bakes an unprovable
    // claim parameter into the callee: a forwarded proj.cast claim reaches this
    // point only after its projection resolved, its proof was inherited from the
    // input, and the identity cast folded away.
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
    if (failed(requireProvenClaimOperands(op, op.getArguments(), rewriter)))
      return failure();

    DemandFrame frame(op.getLoc());

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
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

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

    DemandFrame frame(iface->getLoc());

    // try to compute specialized result types; inference failure defers this op
    SmallVector<Type> specializedTypes;
    if (failed(iface.inferReturnTypes(iface->getContext(), iface->getLoc(),
                                      iface->getOperands(),
                                      iface->getAttrDictionary(),
                                      iface->getPropertiesStorage(),
                                      iface->getRegions(), specializedTypes)))
      return rewriter.notifyMatchFailure(iface, "cannot infer result types from operands");

    // the arity of results must match
    if (specializedTypes.size() != iface->getNumResults())
      return rewriter.notifyMatchFailure(iface, "specialized result type count mismatch");

    // The inferred types are written directly. The participation gate above runs
    // this pattern only while some result is still non-ground, and it reports
    // "result types unchanged" once inference reaches a fixed point, so it
    // cannot spin on its own; a non-confluent interaction with a sibling pattern
    // is caught by the instantiate-monomorphs rewrite budget, which fails loudly
    // rather than livelocking.

    // check if anything actually changes
    if (llvm::equal(iface->getResultTypes(), specializedTypes))
      return rewriter.notifyMatchFailure(iface, "result types unchanged");

    // mutate result types in-place
    rewriter.modifyOpInPlace(iface, [&] {
      for (auto [result, newType] : llvm::zip(iface->getResults(), specializedTypes))
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

    AttrTypeReplacer replacer =
        makeTypeReplacerFromSubstitution(evidence.toTypeMap(), /*module=*/ModuleOp());
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

    DemandFrame frame(op->getLoc());

    AttrTypeReplacer replacer;
    replacer.addReplacement([&](Type t) -> std::optional<Type> {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj)) return std::nullopt;
      auto resolved = resolver.resolveProjectionType(proj, rewriter);
      if (failed(resolved)) {
        recordResolverProjectionMiss(Type(proj));
        return std::nullopt;
      }
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

/// Inherits a proven proof onto a `trait.proj.cast` result whose input is a
/// proven claim naming the same trait application.
///
/// Once the projections inside a forwarded claim are resolved, a proj.cast can
/// be left with a proven input claim and an unproven result claim that name the
/// same trait application, differing only in the proof annotation. This arises
/// when the frontend forwards a generic callable into a projection-spelled
/// bound: the cast's result claim is unproven by construction (see the
/// ProjCastOp description in TraitOps.td), and the projection it carried later
/// normalizes to the same concrete trait application the input already proves.
///
/// A proj.cast never changes which impl proves a claim -- its claim operand
/// only witnesses the projection equality relating input and result. So when
/// the input and result trait applications coincide, they are the same logical
/// claim and must share the same proof. Retyping the result in place to the
/// input's proven type lets the folder collapse the now-identity cast, after
/// which every consumer sees the proven claim. The rewrite is monotone: it only
/// ever turns an unproven result into a proven one.
///
/// A pure folder cannot do this because an MLIR folder may not change a result
/// type; hence this pattern followed by the identity fold.
///
/// Inheriting from the input SSA type is deliberate and cannot be replaced by
/// memo-based proof propagation: frontend-emitted trait.witness proofs are never
/// entered into the resolver's proof memo, so the input value's own type is the
/// only place that proof is recorded. The pattern trusts trait-application
/// equality and does not re-derive the cast's claim-operand justification;
/// soundness is anchored at the proof producers, and per-projection consistency
/// of the cast is enforced separately by the ProjCastOp verifier.
struct InheritProjCastProofPattern : public OpRewritePattern<ProjCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ProjCastOp op,
                                PatternRewriter &rewriter) const override {
    auto inputClaim = dyn_cast<ClaimType>(op.getInput().getType());
    auto resultClaim = dyn_cast<ClaimType>(op.getResult().getType());
    if (!inputClaim || !resultClaim)
      return failure();

    // Guards, all required: input proven, result unproven, and both name the
    // same trait application (hence the same logical claim).
    if (!inputClaim.isProven() || resultClaim.isProven())
      return failure();
    if (inputClaim.getTraitApplication() != resultClaim.getTraitApplication())
      return failure();

    rewriter.modifyOpInPlace(op, [&] { op.getResult().setType(inputClaim); });
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

  // The census is written on every exit from here on, so a run that fails
  // mid-stage still reports what it observed. It is declared before the span
  // below so that the span closes first.
  //
  // This is where the stage ends as far as the ledger is concerned. The pass
  // that wraps it goes on to erase the polymorphs and materialize nominal
  // monomorphs with no sink installed, so what that sweep mints is outside the
  // population by construction.
  auto census = llvm::scope_exit([&] {
    resolver->getDemandLedger().reportServedDrainableKeys(module);
    if (DemandLedger::isRecordingEnabled()) {
      resolver->getDemandLedger().dumpCensus();
      resolver->reportRecordedFacts();
    }
  });

  // The resolver was moved out of the sub-phase that built it, so its ledger is
  // reinstalled here to span this sub-phase's driver and leftover walks. Both
  // spans append to the one ledger the census reports.
  DemandLedgerScope recording(resolver->getDemandLedger());

  // rewrite trait.func.call and trait.method.call, prove claim producers
  // (allege, derive, project), resolve projections, and monomorphize any
  // generic op whose results become monomorphic
  RewritePatternSet patterns(ctx);
  patterns.add<ProveClaimResultPattern>(ctx, *resolver, /*allegeOnly=*/false);
  patterns.add<MonomorphizeResultTypesPattern>(ctx);
  patterns.add<FuncCallOpLowering>(ctx, *resolver);
  patterns.add<MethodCallOpLowering>(ctx, *resolver);
  patterns.add<ResolveProjectionsPattern>(ctx, *resolver);
  patterns.add<PropagateProofsPattern>(ctx, *resolver);
  patterns.add<InheritProjCastProofPattern>(ctx);

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
  if (failed(applyPatternsGreedilyAndReport(module, std::move(patterns), config,
                                            "instantiate-monomorphs")))
    return module.emitError(
        "instantiate-monomorphs did not converge: rewrite budget exceeded, "
        "which indicates a non-confluent pattern pair cycling on a type "
        "spelling");

  // Assert that no op produced an unproven monomorphic claim that escaped
  // proving. Keying this check on the result type rather than on the set of
  // claim-producing ops makes it total over producers: an op whose claims the
  // patterns above fail to discharge is an error here, never a silent gap. The
  // whole result type is walked, so a claim nested inside an aggregate is caught
  // too, not only a claim that is the root type. Trait infrastructure regions
  // are templates and keep their unproven claims.
  bool hasLeftovers = false;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return WalkResult::skip();
    for (Value result : op->getResults()) {
      result.getType().walk([&](Type sub) {
        auto claim = dyn_cast<ClaimType>(sub);
        if (!claim || claim.isProven() || !claim.isMonomorphic())
          return;
        hasLeftovers = true;
        op->emitError() << "unproven monomorphic claim " << claim
                        << " after instantiate-monomorphs";
      });
    }
    return WalkResult::advance();
  });
  if (hasLeftovers) return failure();

  // Reject each concrete-base projection that survived resolution. Walking the
  // result and block-argument types of every non-infrastructure op (operand
  // types are SSA-determined by their producers, so they are covered where those
  // producers are visited), this reports any projection whose base is not
  // symbolic, attributing it to the carrying op ahead of the legalization
  // failure that leftover projection then triggers, instead of leaving that
  // failure the only clue. Projections over still-symbolic bases live only in
  // templates and are left alone; a still-polymorphic template function is not
  // yet instantiated, so its ground projections (over a concrete base nested in
  // an otherwise generic body) resolve when it is cloned for a concrete instance
  // and its whole subtree is skipped.
  bool sawUnresolvedProjection = false;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<TraitOp, ImplOp, ProofOp>(op))
      return WalkResult::skip();
    if (auto func = dyn_cast<func::FuncOp>(op))
      if (isPolymorphicType(Type(func.getFunctionType())))
        return WalkResult::skip();
    auto report = [&](Type root) {
      root.walk([&](Type sub) {
        auto proj = dyn_cast<ProjectionType>(sub);
        if (!proj || isPolymorphicType(proj))
          return;
        op->emitError() << "unresolved projection " << proj
                        << " after instantiate-monomorphs";
        sawUnresolvedProjection = true;
      });
    };
    for (Type t : op->getResultTypes())
      report(t);
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (Value arg : b.getArguments())
          report(arg.getType());
    return WalkResult::advance();
  });
  if (sawUnresolvedProjection)
    return failure();

  DemandRecordingSuspension verifying;
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

  // Materialize the monomorphic symbol definitions the type sweep below will
  // reference.  This runs while the generic templates and concrete type
  // arguments are still present, because the sweep only mangles references to
  // their monomorphic names -- names from which the arguments cannot be
  // recovered -- so every minted monomorphic symbol must have its definition
  // created here first.
  for (Dialect *dialect : ctx->getLoadedDialects())
    if (auto *iface = dialect->getRegisteredInterface<MonomorphizationInterface>())
      iface->materializeMonomorphs(module);

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
