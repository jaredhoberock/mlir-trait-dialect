// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include "TraitOps.hpp"
#include "Trait.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/bit.h>
#include <llvm/Support/Format.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <cinttypes>

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
  void report(StringRef driver, unsigned round, int64_t budget) const {
    llvm::errs() << stageRecordRewritesPrefix << " driver=" << driver
                 << " round=" << round
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

/// Runs one greedy pattern driver over `root` in `round`, reporting the rewrite
/// events it raised under the name `driver` and saying through `changed`
/// whether it rewrote anything at all.
///
/// No listener is installed unless the counts will be reported, so a run that
/// nobody is counting costs exactly what it did before; the driver answers
/// whether it changed the IR on its own.
LogicalResult applyPatternsGreedilyAndReport(Operation *root,
                                             RewritePatternSet &&patterns,
                                             GreedyRewriteConfig config,
                                             StringRef driver, unsigned round,
                                             bool *changed = nullptr) {
  RewriteEventCounts events;
  bool reporting = DemandLedger::isRecordingEnabled();
  if (reporting)
    config.setListener(&events);

  LogicalResult result =
      applyPatternsGreedily(root, std::move(patterns), config, changed);

  if (reporting)
    events.report(driver, round, config.getMaxNumRewrites());
  return result;
}

/// The rewrite budget a driver over `module` runs under.
///
/// Bounding the total rewrite count makes a non-confluent pattern pair fail
/// loudly instead of livelocking. The driver's own iteration limit cannot catch
/// a livelock: two patterns that keep undoing each other's in-place type
/// rewrites hold the worklist non-empty within one iteration. The bound scales
/// with input size; a legitimate run rewrites each op a small bounded number of
/// times as its types refine, so a run that reaches the bound is cycling.
int64_t rewriteBudgetFor(ModuleOp module) {
  int64_t opCount = 0;
  module.walk([&](Operation *) { ++opCount; });
  return opCount * 1024 + 4096;
}

} // namespace

LogicalResult convertToTrait(ModuleOp module, unsigned round,
                             bool *changed = nullptr) {
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);

  // collect patterns from participating dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateConvertToTraitPatterns(patterns);
  }

  GreedyRewriteConfig config;
  config.setMaxNumRewrites(rewriteBudgetFor(module));

  // apply patterns
  if (failed(applyPatternsGreedilyAndReport(module, std::move(patterns), config,
                                            "convert-to-trait", round, changed)))
    return module.emitError(
        "convert-to-trait did not converge: rewrite budget exceeded, which "
        "indicates a non-confluent pattern pair cycling on a type spelling");

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
      // This check runs before the stage builds a resolver, so it holds no
      // memo and derives what it needs itself.
      if (failed(verifyAndRecordProof(claim.asUnproven(), claim, module, bindings,
                                      DemandOrigin::ProofRecording,
                                      /*memo=*/nullptr, errFn)))
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

/// The replacer the agreement check compares against: the same rewrite built as
/// a substitution over the whole proof memo. Nothing is built when the check is
/// disarmed, which is what keeps the whole-memo copy off every other run.
static std::optional<AttrTypeReplacer>
makeSubstitutionCheckReplacer(const ImplResolver &resolver) {
  if (!DemandLedger::isPostconditionEnabled())
    return std::nullopt;
  return makeTypeReplacerFromSubstitution(
      resolver.buildClaimSubstitutionFromMemo().toTypeMap(),
      /*module=*/ModuleOp());
}

/// Reports every position of `op` that `queried` and `substituted` respell
/// differently.
///
/// The two constructions walk a type differently: a substitution hands every
/// generic type its own specialization step and stops the walk there, while a
/// memo lookup walks the type structurally and answers only for claims. The
/// claim substitution binds no generic type, so that specialization step has
/// nothing to apply, and only the structural walk reaches a claim nested inside
/// a generic type. This states that reasoning where it can be contradicted, on
/// every position the stage respells. A disagreement is a gap in this
/// reasoning, never a fault in the program being compiled, so it goes to the
/// census channel; the whole spelling on each side is printed because which
/// component of a type moved is the question a reader is left with.
static void reportRespellingDisagreements(AttrTypeReplacer &queried,
                                          AttrTypeReplacer &substituted,
                                          Operation *op) {
  auto report = [&](const Twine &position, auto before, auto byLookup,
                    auto bySubstitution) {
    llvm::errs() << demandCensusRespellingDisagreementPrefix
                 << " op=" << op->getName() << " at=" << op->getLoc()
                 << " position=" << position << " before=" << before
                 << " by-lookup=" << byLookup
                 << " by-substitution=" << bySubstitution << "\n";
  };

  auto compareType = [&](const Twine &position, Type before) {
    Type byLookup = queried.replace(before);
    Type bySubstitution = substituted.replace(before);
    if (byLookup != bySubstitution)
      report(position, before, byLookup, bySubstitution);
  };

  for (auto [index, type] : llvm::enumerate(op->getResultTypes()))
    compareType("result " + Twine(index), type);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (BlockArgument argument : block.getArguments())
        compareType("block argument " + Twine(argument.getArgNumber()),
                    argument.getType());
  for (NamedAttribute attribute : op->getAttrs()) {
    Attribute byLookup = queried.replace(attribute.getValue());
    Attribute bySubstitution = substituted.replace(attribute.getValue());
    if (byLookup != bySubstitution)
      report("attribute " + attribute.getName().getValue(),
             attribute.getValue(), byLookup, bySubstitution);
  }
}

/// Respells throughout `root` every claim `resolver` has recorded a proof for,
/// and returns how many positions of `root` that sweep moved.
///
/// The replacer's recursive entry point is this walk, so driving the walk here
/// costs nothing extra and is what lets an op the sweep respelled be told from
/// one it left alone. A position is a result type, a block-argument type, or the
/// attribute dictionary of one op; an op counts once however many of its
/// positions moved. The count is what says whether this sweep wrote anything,
/// so it is taken whether or not anyone is reading the report.
///
/// Each op is named while it is visited, so a demand raised under the sweep is
/// attributed to the op carrying the type rather than to the whole module.
///
/// The sweep records no proof of its own, which is the precondition the
/// replacer it holds across the whole walk asserts.
///
/// `anchor`, when given, receives where one op the sweep moved was written, for
/// a diagnostic that must name somewhere the round's work landed. A location
/// rather than the op, because the instantiation that follows the sweep may
/// erase what the sweep just respelled.
static uint64_t respellProvenClaimsInPlace(const ImplResolver &resolver,
                                           Operation *root, unsigned round,
                                           std::optional<Location> *anchor = nullptr) {
  size_t recordedProofs = resolver.getRecordedProofCount();
  if (recordedProofs == 0) return 0;
  AttrTypeReplacer replacer = resolver.makeProvenClaimReplacer();
  std::optional<AttrTypeReplacer> substituted =
      makeSubstitutionCheckReplacer(resolver);

  uint64_t opsRespelled = 0;
  uint64_t positionsRespelled = 0;

  root->walk([&](Operation *op) {
    DemandFrame frame(op->getLoc());

    if (substituted)
      reportRespellingDisagreements(replacer, *substituted, op);

    SmallVector<Type, 8> before;
    auto eachTypePosition = [&](llvm::function_ref<void(Type)> visit) {
      for (Type type : op->getResultTypes())
        visit(type);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            visit(argument.getType());
    };
    eachTypePosition([&](Type type) { before.push_back(type); });
    DictionaryAttr attributesBefore = op->getAttrDictionary();

    replacer.replaceElementsIn(op,
                               /*replaceAttrs=*/true,
                               /*replaceLocs=*/false,
                               /*replaceTypes=*/true);

    uint64_t movedHere = op->getAttrDictionary() == attributesBefore ? 0 : 1;
    size_t position = 0;
    eachTypePosition([&](Type type) {
      if (type != before[position++])
        ++movedHere;
    });
    positionsRespelled += movedHere;
    if (movedHere) {
      ++opsRespelled;
      if (anchor)
        *anchor = op->getLoc();
    }
  });

  // A sweep records no proof, so the count of facts does not move for it; what
  // it moves is the module's spelling of them, which is what proof derivation
  // reads. A sweep that respelled nothing leaves every derivation reading the
  // module it read.
  if (positionsRespelled != 0)
    resolver.noteRespelling();

  if (DemandLedger::isRecordingEnabled())
    llvm::errs() << stageRecordRespellingPrefix
                 << " round=" << round
                 << " bindings=" << recordedProofs
                 << " ops=" << opsRespelled
                 << " positions=" << positionsRespelled << "\n";
  return positionsRespelled;
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
  if (failed(convertToTrait(module, /*round=*/0)))
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
                                              "resolve-impls", /*round=*/0)))
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
  // may itself contain a claim that was just proven.  Respell all
  // unproven claims in their proven forms so that downstream instantiation
  // sees consistent types.
  respellProvenClaimsInPlace(resolver, module, /*round=*/0);

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
    TypeRange types, ModuleOp module, ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  for (Type ty : types) {
    Type rewritten = apply(ty);
    // Closing a call substitution needs the stage's resolver, so this walk
    // runs nowhere else and its demand is the stage's.
    if (failed(recordProofBindingsIn(rewritten, module, evidenceBindings,
                                     DemandOrigin::ProofRecording, memo, err)))
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

    if (failed(discoverEvidenceBindings(operandTypes, module,
                                        &resolver.getDerivationMemo(), err)))
      return failure();
    if (failed(discoverEvidenceBindings(resultTypes, module,
                                        &resolver.getDerivationMemo(), err)))
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
  auto subst = op.buildParameterSpecialization(module,
                                              &resolver.getDerivationMemo());
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

  auto callee = op.getOrSpecializeCallee(rewriter, *subst,
                                        &resolver.getDerivationMemo());
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

/// What one round did, which is what says whether another round has anything to
/// do.
///
/// A round that wrote nothing minted no fact, found no demand nothing had seen
/// and rewrote nothing, so the round after it would repeat it exactly. Asking
/// again about a demand an earlier round already asked about is not writing: a
/// loop that ran on questions rather than answers would run until its bound.
struct RoundWork {
  /// Whether the bridge into trait vocabulary rewrote anything.
  bool bridged = false;
  /// Demands put to impl selection.
  size_t collected = 0;
  /// How those demands were settled: resolved, refused on the arm no later
  /// resolution overturns, or left for a later round.
  uint64_t served = 0;
  uint64_t refused = 0;
  uint64_t deferred = 0;
  /// Ops impl selection inserted serving them.
  uint64_t insertedServingDemands = 0;
  /// Type positions the round's commit respelled.
  uint64_t respelled = 0;
  /// Whether the instantiation driver rewrote anything.
  bool instantiated = false;
  /// How the collected demands split by the way the lookup missed on them, how
  /// many were declined by an engine that names no way at all, and how many
  /// carry more than one of the lookup's arms.
  uint64_t noCandidateImpl = 0;
  uint64_t multipleCandidateImpls = 0;
  uint64_t otherArms = 0;
  uint64_t withoutArm = 0;
  uint64_t ambiguousArms = 0;
  /// Refusals the round's flush forgot and kept, and what became of the ones
  /// the round before it forgot.
  ImplResolver::RefusalCounts refusals;
  /// Impls the instantiation driver asked to have generated mid-run.
  uint64_t midDriverGeneration = 0;
  /// Whether impl selection minted a fact anywhere in the round.
  bool mintedFacts = false;
  /// Whether the drain grew after the round had already collected from it, so
  /// that a demand raised late in the round has had no round put it to
  /// selection.
  bool drainGrewAfterCollect = false;

  bool wrote() const {
    return bridged || served || insertedServingDemands || respelled ||
           instantiated || mintedFacts || drainGrewAfterCollect;
  }
};

/// Counts the ops one round's own resolution inserts.
///
/// Resolution under a pattern driver reaches that driver's worklist through the
/// rewriter's listener. A round resolves between its drivers and then runs the
/// next one over the whole module, which reaches an op inserted here without it
/// having been enqueued; the count is what the listener is for.
struct RoundInsertionCounts : public OpBuilder::Listener {
  void notifyOperationInserted(Operation *, OpBuilder::InsertPoint) override {
    ++inserted;
  }

  uint64_t inserted = 0;
};

/// The demands `ledger` holds that no round has settled and that the facts have
/// moved under since a round last asked about them.
///
/// A demand leaves the drain for good when nothing a later round could ask
/// would settle it differently: impl selection resolved it, or refused it on
/// the arm no later resolution overturns. `drained` holds those. One selection
/// could not serve yet stays on the drain, and `attempted` carries the fact
/// epoch it was last put to selection at, so a round asks about it again
/// exactly where selection has minted something since -- which is the only
/// thing that can make the answer differ, and the only thing that keeps asking
/// again from asking the same question forever.
static SmallVector<Type>
collectUndrainedDemands(const DemandLedger &ledger,
                        const DenseSet<Type> &drained,
                        const DenseMap<Type, uint64_t> &attempted,
                        uint64_t epoch) {
  SmallVector<Type> collected;
  for (Type demand : ledger.getDrainableDemands()) {
    if (drained.contains(demand))
      continue;
    auto it = attempted.find(demand);
    if (it != attempted.end() && it->second == epoch)
      continue;
    collected.push_back(demand);
  }
  return collected;
}

/// Records in `work` how `collected` splits by the way the lookup missed.
///
/// A demand no impl binds is one a generator can supply; a demand several impls
/// bind is one the premise partition must choose among, and generating for it
/// would add a candidate to an application that already has too many. Both are
/// served the same way -- by the impl selection that decides which case it is --
/// so the split is what the round reports rather than what it routes on.
static void splitCollectedDemands(const DemandLedger &ledger,
                                  ArrayRef<Type> collected, RoundWork &work) {
  for (Type demand : collected) {
    unsigned arms = ledger.getDrainableArms(demand);
    // The lookup is the one engine that declines a demand in more than one way,
    // and it names the way it declined. Its arms accumulate over the whole
    // stage, across the module changes the rounds themselves make, so one
    // demand can carry the arm it missed on before an impl was generated for it
    // and the arm it misses on after. Serving does not route on the arm, so a
    // demand carrying two of them is reported as carrying two.
    if (llvm::popcount(arms) > 1)
      ++work.ambiguousArms;
    if (arms & (1u << unsigned(LookupMissReason::MultipleCandidateImpls)))
      ++work.multipleCandidateImpls;
    else if (arms & (1u << unsigned(LookupMissReason::NoCandidateImpl)))
      ++work.noCandidateImpl;
    else if (arms)
      ++work.otherArms;
    else
      ++work.withoutArm;
  }
}

/// Puts every demand in `collected` to impl selection, which generates the impl
/// the demand needs when none binds its application and partitions the
/// candidates when several do, and records what each attempt settled.
///
/// A demand selection resolved or refused for good leaves the drain; one it
/// could not serve yet stays, against the epoch it was asked at.
static void serveCollectedDemands(ImplResolver &resolver,
                                  ArrayRef<Type> collected, OpBuilder &builder,
                                  DenseSet<Type> &drained,
                                  DenseSet<Type> &served,
                                  DenseMap<Type, uint64_t> &attempted,
                                  RoundWork &work) {
  for (Type demand : collected) {
    // Every engine whose declining leaves a demand standing declines a
    // monomorphic projection and nothing else, so the drain holds projections
    // and this is total over what it holds.
    auto projection = dyn_cast<ProjectionType>(demand);
    assert(projection &&
           "a drainable demand is a projection an engine left spelled");

    // The epoch is read per demand rather than once per round: serving one
    // demand mints facts the demands after it in this batch are resolved
    // under, so a demand asked about before that is one the next round asks
    // about again.
    attempted[demand] = resolver.getFactEpoch();
    switch (resolver.serveDemand(projection, builder)) {
    case ImplResolver::DemandDisposition::Served:
      ++work.served;
      drained.insert(demand);
      served.insert(demand);
      break;
    case ImplResolver::DemandDisposition::Refused:
      ++work.refused;
      drained.insert(demand);
      break;
    case ImplResolver::DemandDisposition::Deferred:
      ++work.deferred;
      break;
    }
  }
}

/// Writes one round's line, naming what it did and the facts it left behind.
static void reportRound(unsigned round, const RoundWork &work,
                        const ImplResolver &resolver) {
  llvm::errs() << stageRecordRoundPrefix << " index=" << round
               << " bridged=" << (work.bridged ? "yes" : "no")
               << " collected=" << work.collected
               << " no-candidate-impl=" << work.noCandidateImpl
               << " multiple-candidate-impls=" << work.multipleCandidateImpls
               << " other-arms=" << work.otherArms
               << " without-arm=" << work.withoutArm
               << " ambiguous-arms=" << work.ambiguousArms
               << " served=" << work.served
               << " declined=" << work.refused + work.deferred
               << " deferred=" << work.deferred
               << " inserted-serving-demands=" << work.insertedServingDemands
               << " respelled-positions=" << work.respelled
               << " refusals-forgotten=" << work.refusals.forgotten
               << " refusals-kept=" << work.refusals.kept
               << " refusals-overturned=" << work.refusals.overturned
               << " refusals-re-earned=" << work.refusals.reEarned
               << " mid-driver-generation=" << work.midDriverGeneration
               << " instantiated=" << (work.instantiated ? "yes" : "no")
               << llvm::format(" digest=0x%016" PRIx64,
                               resolver.getRecordedFactsDigest())
               << "\n";
}

/// Checks that impl selection left nothing part-way done.
///
/// Selection is entered at round zero, at the round's own generation step, and
/// by the patterns the instantiation driver runs. A reader of its facts between
/// those points must find every application it opened closed and every proof it
/// recorded naming a proof the module defines, because a fact read part-way
/// through is one that is not yet a fact.
static void checkResolutionBoundary(const ImplResolver &resolver) {
  assert(resolver.isQuiescent() &&
         "impl selection must not be part-way through an application at a "
         "boundary between the stage's steps");
  // Reading the second half costs a symbol table over a module every round
  // rewrites, so it is asked where the cross-checks are armed rather than on
  // every compile.
  assert((!DemandLedger::isPostconditionEnabled() ||
          resolver.recordsOnlyRealizedProofs()) &&
         "every proof recorded at a boundary between the stage's steps must "
         "name a proof the module defines");
}

} // end namespace

LogicalResult instantiateMonomorphs(ModuleOp module) {
  // Round zero: resolve the impls the module already spells and respell the
  // claims they prove, before any round asks for an impl that is missing.
  auto resolver = resolveImpls(module);
  if (failed(resolver))
    return failure();
  checkResolutionBoundary(*resolver);

  MLIRContext* ctx = module.getContext();

  // The demands the rounds below settled and the ones they served. A demand is
  // settled when nothing a later round could ask would answer differently, so
  // the served demands are a subset: a demand refused on the arm no later
  // resolution overturns is settled and unserved. The stage-exit checks read
  // both -- the served set tells a demand the stage answered from one the
  // drainability rule over-admitted, and the difference is what must still be
  // spelled for something to report.
  DenseSet<Type> drained;
  DenseSet<Type> served;
  // The fact epoch each unsettled demand was last put to selection at, which is
  // what says whether asking again could answer differently.
  DenseMap<Type, uint64_t> attempted;

  // The census is written on every exit from here on, so a run that fails
  // mid-stage still reports what it observed. It is declared before the span
  // below so that the span closes first.
  //
  // This is where the stage ends as far as the ledger is concerned. The pass
  // that wraps it goes on to erase the polymorphs and materialize nominal
  // monomorphs with no sink installed, so what that sweep mints is outside the
  // population by construction.
  auto census = llvm::scope_exit([&] {
    resolver->getDemandLedger().reportServedDrainableKeys(module, served);
    if (DemandLedger::isRecordingEnabled()) {
      resolver->getDemandLedger().dumpCensus();
      resolver->reportRecordedFacts();
    }
  });

  // The resolver was moved out of the sub-phase that built it, so its ledger is
  // reinstalled here to span this sub-phase's rounds and leftover walks. Both
  // spans append to the one ledger the census reports.
  DemandLedgerScope recording(resolver->getDemandLedger());

  // A round forgets the refusals a later resolution could answer differently,
  // bridges into trait vocabulary, takes the demands nothing has settled off
  // the drain, puts them to impl selection, commits what selection proved to
  // the module's spellings, and only then instantiates. Rounds run until one of
  // them writes nothing, at which point the round after it would repeat it.
  //
  // The flush leads because everything after it asks questions: a round asking
  // under a negative an earlier round recorded would be told what was true
  // before the impls this stage has generated since existed.
  //
  // Only the demands an engine left standing are collected here. The
  // obligations impl selection raises proving one claim are resolved on the
  // same stack that raised them and never reach the drain, and the projections
  // instantiation meets are served inside the driver by the patterns that meet
  // them; both are pinned by the leftover walks at the end of the stage.
  //
  // Each round's work is bounded by the module and a round that finds nothing
  // ends the loop, so the count of rounds is the depth of the chain of impls
  // the module needs generated. A module whose rounds keep finding work is
  // cycling, and this bound is what makes that loud rather than endless.
  constexpr unsigned maxRounds = 64;
  unsigned round = 0;
  // Whether anything has written to the module since the bridge last ran and
  // since the commit last swept it. A step whose input has not moved since it
  // last ran would produce what it produced then, which for both of these is
  // nothing.
  bool writtenSinceBridge = true;
  bool writtenSinceSweep = false;
  size_t proofsAtSweep = resolver->getRecordedProofCount();
  // What the last two rounds did, and where the last commit moved something,
  // for the report the round bound owes a reader.
  SmallVector<std::pair<unsigned, RoundWork>, 2> lastRounds;
  std::optional<Location> lastRespelled;
  for (bool wrote = true; wrote;) {
    if (++round > maxRounds) {
      // What the last two rounds did is what says which work is coming back
      // round, so it is written whether or not anyone asked for the record.
      for (const auto &[index, past] : lastRounds)
        reportRound(index, past, *resolver);
      InFlightDiagnostic diagnostic =
          emitError(lastRespelled.value_or(module.getLoc()));
      return diagnostic
             << "instantiate-monomorphs did not converge: the stage ran its "
                "rounds to the round bound, which indicates a round writing "
                "work back for the next one to find";
    }

    RoundWork work;
    uint64_t epochAtRoundHead = resolver->getFactEpoch();

    // FLUSH. Every refusal a later resolution could answer differently is
    // forgotten here, so that the questions the rest of the round asks are
    // asked against the facts as they now stand.
    work.refusals = resolver->forgetRetriableRefusals();

    // BRIDGE. The patterns that lift another dialect's vocabulary into trait
    // claims run whenever something has written to the module since they last
    // ran, because that writing may have created the ops they lift.
    if (writtenSinceBridge) {
      if (failed(convertToTrait(module, round, &work.bridged)))
        return failure();
      writtenSinceBridge = false;
      writtenSinceSweep |= work.bridged;
    }

    // COLLECT.
    SmallVector<Type> collected = collectUndrainedDemands(
        resolver->getDemandLedger(), drained, attempted,
        resolver->getFactEpoch());
    work.collected = collected.size();
    splitCollectedDemands(resolver->getDemandLedger(), collected, work);
    size_t drainAtCollect =
        resolver->getDemandLedger().getDrainableDemands().size();

    // GENERATE.
    {
      RoundInsertionCounts insertions;
      OpBuilder builder(ctx);
      builder.setListener(&insertions);
      builder.setInsertionPointToEnd(module.getBody());
      serveCollectedDemands(*resolver, collected, builder, drained, served,
                            attempted, work);
      work.insertedServingDemands = insertions.inserted;
    }
    writtenSinceBridge |= work.insertedServingDemands != 0;
    writtenSinceSweep |= work.insertedServingDemands != 0;

    // COMMIT. Every claim the stage has proved is respelled in its proven form
    // throughout the module, so the round that follows reads one spelling of
    // each claim wherever it appears.
    //
    // The sweep rewrites a claim where the module spells one the proof memo
    // answers for, so a module nothing has written to since the last sweep,
    // under a memo that has not grown since, has nothing left for it to move.
    if (writtenSinceSweep || resolver->getRecordedProofCount() != proofsAtSweep) {
      work.respelled =
          respellProvenClaimsInPlace(*resolver, module, round, &lastRespelled);
      proofsAtSweep = resolver->getRecordedProofCount();
      writtenSinceSweep = false;
      writtenSinceBridge |= work.respelled != 0;
    }

    checkResolutionBoundary(*resolver);

    // INSTANTIATE. Rewrite trait.func.call and trait.method.call, prove claim
    // producers (allege, derive, project), resolve projections, and monomorphize
    // any generic op whose results become monomorphic.
    RewritePatternSet patterns(ctx);
    patterns.add<ProveClaimResultPattern>(ctx, *resolver, /*allegeOnly=*/false);
    patterns.add<MonomorphizeResultTypesPattern>(ctx);
    patterns.add<FuncCallOpLowering>(ctx, *resolver);
    patterns.add<MethodCallOpLowering>(ctx, *resolver);
    patterns.add<ResolveProjectionsPattern>(ctx, *resolver);
    patterns.add<InheritProjCastProofPattern>(ctx);

    // collect instantiate-monomorphs patterns from other dialects
    for (Dialect *d : ctx->getLoadedDialects()) {
      if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
        iface->populateInstantiateMonomorphsPatterns(patterns);
    }

    GreedyRewriteConfig config;
    config.setMaxNumRewrites(rewriteBudgetFor(module));

    {
      // Generating an impl is a round's own work. One generated while the
      // driver runs is a fact the run's earlier rewrites could not see, so what
      // is left of that is counted where it happens.
      ImplGenerationTally midDriverGeneration(*resolver);
      // XXX TODO: a freeze stands over this span only where the environment
      // asks for one, which is how the freeze is reached while nothing arms it
      // for real. It replaces the tally outright once the driver stops asking.
      std::optional<ImplGenerationFreeze> freeze;
      if (isInstantiationFreezeRequested())
        freeze.emplace(*resolver, "the instantiation driver");
      LogicalResult instantiated = applyPatternsGreedilyAndReport(
          module, std::move(patterns), config, "instantiate-monomorphs", round,
          &work.instantiated);
      work.midDriverGeneration = midDriverGeneration.getAsks();
      if (failed(instantiated))
        return module.emitError(
            "instantiate-monomorphs did not converge: rewrite budget exceeded, "
            "which indicates a non-confluent pattern pair cycling on a type "
            "spelling");
    }
    writtenSinceBridge |= work.instantiated;
    writtenSinceSweep |= work.instantiated;

    checkResolutionBoundary(*resolver);

    // A demand raised after this round collected has had no round put it to
    // selection, and a fact minted anywhere in the round is one the round
    // before could not have seen; either is work for a round after this one.
    work.drainGrewAfterCollect =
        resolver->getDemandLedger().getDrainableDemands().size() >
        drainAtCollect;
    work.mintedFacts = resolver->getFactEpoch() != epochAtRoundHead;

    wrote = work.wrote();
    if (DemandLedger::isRecordingEnabled())
      reportRound(round, work, *resolver);
    if (lastRounds.size() == 2)
      lastRounds.erase(lastRounds.begin());
    lastRounds.emplace_back(round, work);
  }

  // Every demand a round took off the drain was one it undertook to settle, so
  // at the end of the stage each is served or left for the walks below to
  // report. A demand taken and dropped is one nothing downstream would mention.
  if (failed(resolver->getDemandLedger().checkDrainedKeysSettled(module, drained,
                                                                 served)))
    return failure();

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
