// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include "TraitOps.hpp"
#include "Trait.hpp"
#include "TraitTypes.hpp"
#include <cstdlib>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

namespace mlir::trait {

//===----------------------------------------------------------------------===//
// convertToTrait
//===----------------------------------------------------------------------===//

namespace {

/// Tracks rewritten roots and enforces the total greedy rewrite budget.
struct RewriteEventCounts : public RewriterBase::Listener {
  using RewriterBase::Listener::notifyOperationReplaced;

  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint) override {
    noteWritten(op);
  }
  void notifyOperationModified(Operation *op) override {
    noteWritten(op);
  }
  void notifyOperationReplaced(Operation *op, ValueRange) override {
    noteWritten(op);
  }
  void notifyOperationErased(Operation *op) override {
    liveWritten.erase(op);
  }

  void notifyPatternEnd(const Pattern &, LogicalResult status) override {
    if (succeeded(status))
      ++applications;
  }

  /// Says where a driver may record what it writes. Without it nothing is
  /// recorded, which is what a caller that reads no record wants.
  void recordWritesUnder(Block *body) { moduleBody = body; }

  /// The module-level ops a rewrite reached since the last drain and that still
  /// stand, in the module's own order. Draining starts a fresh record.
  ///
  /// A rewrite is recorded against the module-level op containing it because
  /// that is the unit a further pass over the module must reconsider: a value
  /// the rewrite produced is read by its neighbours, and under `ExistingOps`
  /// strictness a neighbour is reconsidered only if it is handed to the driver.
  /// An op erased since is not here -- the erase notification drops it -- so no
  /// address this answers with has been freed.
  SmallVector<Operation *> takeWrittenModuleLevelOps(Block *body) {
    DenseMap<Operation *, unsigned> position;
    unsigned index = 0;
    for (Operation &op : *body)
      position[&op] = index++;
    SmallVector<Operation *> standing;
    for (Operation *op : writtenOrder)
      if (liveWritten.contains(op) && position.count(op))
        standing.push_back(op);
    llvm::sort(standing, [&](Operation *a, Operation *b) {
      return position[a] < position[b];
    });
    standing.erase(llvm::unique(standing), standing.end());
    writtenOrder.clear();
    liveWritten.clear();
    return standing;
  }

  uint64_t applications = 0;

private:
  /// Records `op`'s module-level ancestor, the op standing directly in the
  /// module's body that contains it.
  void noteWritten(Operation *op) {
    if (!moduleBody)
      return;
    Operation *current = op;
    while (current && current->getBlock() != moduleBody)
      current = current->getParentOp();
    if (!current)
      return;
    if (liveWritten.insert(current).second)
      writtenOrder.push_back(current);
  }

  Block *moduleBody = nullptr;
  SmallVector<Operation *> writtenOrder;
  DenseSet<Operation *> liveWritten;
};

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

/// Whether `op` is a symbol declaration whose own spelling still carries a
/// generic type: a template another dialect owns, specialized per concrete use
/// and cut once its instances are cloned. The generic is read off the
/// declaration's attributes -- its type parameters and the body they parameterize
/// -- so this dialect recognizes such a symbol without naming the dialect that
/// declares it.
static bool isGenericSymbolDeclaration(Operation *op) {
  if (!isa<SymbolOpInterface>(op))
    return false;
  // A region-less declaration parameterizes its body through its attributes; a
  // symbol-table op (one that holds regions) is not one, so its interior is not
  // hidden behind a template shell here.
  if (op->getNumRegions() != 0)
    return false;
  bool generic = false;
  op->getAttrDictionary().walk([&](Type ty) {
    if (isa<GenericTypeInterface>(ty)) {
      generic = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return generic;
}

/// True when `op` is itself a generic template: a trait, impl, or proof
/// declaration, a still-polymorphic function, or any other symbol declaration
/// whose spelling still carries a generic type. Instantiation carries a template
/// to no target; it dies when its concrete instances are cloned. The shell op is
/// a template too, so this answers true for the declaration itself, not only for
/// code nested inside one.
static bool isTemplate(Operation *op) {
  if (isa<TraitOp, ImplOp, ProofOp>(op))
    return true;
  if (auto func = dyn_cast<func::FuncOp>(op))
    return isPolymorphicType(Type(func.getFunctionType()));
  return isGenericSymbolDeclaration(op);
}

/// Appends to `ops` the ops of `root`'s subtree a rewrite driver may reach,
/// `root` itself included: every op outside a template. One pre-order walk skips
/// a template whole -- its shell and its interior. With `includeTemplateShells`
/// a template's shell op is kept while its interior is still skipped, for the
/// one bridge pattern that anchors on a trait declaration.
static void collectRewritableOpsIn(Operation *root, bool includeTemplateShells,
                                   SmallVectorImpl<Operation *> &ops) {
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op)) {
      if (includeTemplateShells)
        ops.push_back(op);
      return WalkResult::skip();
    }
    ops.push_back(op);
    return WalkResult::advance();
  });
}

/// The ops in `module` a rewrite driver may reach, in the module's own order.
static SmallVector<Operation *>
collectRewritableOps(ModuleOp module, bool includeTemplateShells) {
  SmallVector<Operation *> ops;
  for (Operation &op : *module.getBody())
    collectRewritableOpsIn(&op, includeTemplateShells, ops);
  return ops;
}

/// Runs `patterns` greedily over the ops `module` reaches (collectRewritableOps),
/// then over the module-level ops each iteration wrote to, until an iteration
/// writes to none. `ExistingOps` strictness admits nothing a run creates, so a
/// clone is invisible to the iteration that made it and the iteration after is
/// what reaches it; the default strictness would follow a rewritten producer
/// into a region, and `ExistingAndNewOps` would enqueue every op a clone's
/// construction creates. A function no iteration wrote to stands where a
/// previous iteration drove it to a fixed point, and what a pattern reads
/// besides the op itself is the recorded facts, whose movement the round loop
/// outside answers by running the driver again over the whole module; so
/// carrying only the written functions forward is the same fixed point reached
/// proportionally to the work rather than to the module. The rewrite budget
/// spans the whole run: the listener counts applications across iterations, each
/// iteration receives the remainder, and an exhausted remainder or a
/// non-converged iteration fails as one whole-module run does.
static LogicalResult applyPatternsOverReachableOps(ModuleOp module,
                                                   RewritePatternSet &&patterns,
                                                   GreedyRewriteConfig config,
                                                   bool *changed,
                                                   bool includeTemplateShells) {
  FrozenRewritePatternSet frozen(std::move(patterns));
  RewriteEventCounts events;
  events.recordWritesUnder(module.getBody());
  config.setListener(&events);
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  config.setScope(&module.getBodyRegion());
  int64_t budget = config.getMaxNumRewrites();

  bool anyChange = false;
  LogicalResult result = success();
  bool firstIteration = true;
  while (succeeded(result)) {
    SmallVector<Operation *> ops;
    if (firstIteration) {
      ops = collectRewritableOps(module, includeTemplateShells);
    } else {
      // The module-level ops the previous iteration wrote to are the whole of
      // the work left for this one: a clone it minted, and every function a
      // rewrite landed in, whose neighbouring ops read what that rewrite
      // produced.
      for (Operation *op : events.takeWrittenModuleLevelOps(module.getBody()))
        collectRewritableOpsIn(op, includeTemplateShells, ops);
    }
    firstIteration = false;
    if (ops.empty())
      break;
    if (budget >= 0) {
      int64_t remaining = budget - int64_t(events.applications);
      if (remaining < 0) {
        result = failure();
        break;
      }
      config.setMaxNumRewrites(remaining);
    }
    bool iterationChanged = false;
    result = applyOpPatternsGreedily(ops, frozen, config, &iterationChanged);
    anyChange |= iterationChanged;
    if (!iterationChanged)
      break;
  }

  if (changed)
    *changed = anyChange;
  return result;
}

} // namespace

bool isForeign(Operation *op) {
  if (isTemplate(op))
    return true;
  for (Operation *ancestor = op->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp())
    if (isTemplate(ancestor))
      return true;
  return false;
}

LogicalResult convertToTrait(ModuleOp module, bool *changed = nullptr) {
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);

  // collect patterns from participating dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateConvertToTraitPatterns(patterns);
  }

  GreedyRewriteConfig config;
  config.setMaxNumRewrites(rewriteBudgetFor(module));

  // apply patterns. Shells are included: tuple's mapper-trait bridge anchors on
  // a trait declaration, the one place a pattern is handed a template shell.
  if (failed(applyPatternsOverReachableOps(module, std::move(patterns), config,
                                           changed,
                                           /*includeTemplateShells=*/true)))
    return module.emitError(
        "convert-to-trait did not converge: rewrite budget exceeded, which "
        "indicates a non-confluent pattern pair cycling on a type spelling");

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

// The structural half of the acyclicity check, split from the full verify tail
// so it is safe on unverified IR: it reads trait symbols by name and refuses a
// dangling `where`-clause reference through a diagnostic rather than reaching the
// aborting trait accessor. The trait-to-trait edges it walks form the
// `where`-clause dependency graph; a back-edge is a cycle.
LogicalResult verifyAcyclicTraitsStructure(ModuleOp module) {
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

    // The `where` clause is a mandatory property, so verifier-valid IR always
    // carries at least an empty predicate array (a no-`where` trait prints
    // `requirements = #trait<predicate_array[]>`). A null here is malformed
    // input the screen faces before the full verifier -- refuse it rather than
    // dereference the null attribute in the range below.
    PredicateArrayAttr requirements = u.getRequirements();
    if (!requirements)
      return u.emitError("trait carries no `where`-clause requirements array");

    for (Attribute pred : requirements) {
      // Only application requirements form trait-to-trait edges; an equality
      // requirement has no trait head and cannot close a `where`-clause cycle.
      auto app = dyn_cast<TraitApplicationAttr>(pred);
      if (!app)
        continue;
      // Resolve the edge's target by name. A `where` clause naming a trait the
      // module does not define is refused here -- a hostile blob reaches this
      // screen before the full verifier, so this must not reach the aborting
      // accessor.
      FailureOr<TraitOp> v = app.getTrait(module, /*errFn=*/nullptr);
      if (failed(v))
        return u.emitError("trait `where` clause references undefined trait '")
               << app.getTraitName().getValue() << "'";
      // A requirement like @Trait[!trait.proj<@Trait[!S], "Assoc">] is a
      // syntactic self-reference, but not a real cycle: the projection resolves
      // to a concrete type during monomorphization, breaking the edge. Skip it
      // so that traits with bounded associated types (e.g. `type Assoc: Trait`)
      // don't falsely trigger the acyclicity check. A TraitApplicationAttr has
      // no verifier, so its type-argument list can be empty on bytecode input
      // (the text parser requires at least one, but no gate stands before this
      // screen); an empty self-edge carries no projection to break the cycle, so
      // it falls through to the back-edge check below rather than the `.front()`
      // dereference.
      ArrayRef<Type> typeArgs = app.getTypeArgs();
      if (*v == u && !typeArgs.empty() &&
          containsType<ProjectionType>(typeArgs.front()))
        continue;
      if (failed(dfs(*v))) return failure();
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

  return success();
}

LogicalResult verifyAcyclicTraits(ModuleOp module) {
  if (failed(verifyAcyclicTraitsStructure(module)))
    return failure();

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

/// Respells throughout `root` every claim `resolver` has recorded a proof for,
/// and returns how many positions of `root` that sweep moved.
///
/// The replacer's recursive entry point is this walk, so driving the walk here
/// costs nothing extra and is what lets an op the sweep respelled be told from
/// one it left alone. A position is a result type, a block-argument type, or the
/// attribute dictionary of one op. The count is what says whether this sweep
/// wrote anything.
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
                                           Operation *root,
                                           std::optional<Location> *anchor = nullptr) {
  size_t recordedProofs = resolver.getRecordedProofCount();
  size_t recordedImpls = resolver.getRecordedImplCount();
  if (recordedProofs == 0 && recordedImpls == 0) return 0;
  AttrTypeReplacer replacer = resolver.makeProvenClaimReplacer();
  // Beside the proofs it respells, the sweep resolves recorded ground
  // projections, so an interior op stays consistent with an outside value a
  // pattern retyped -- a tuple.make or arith.select whose result the outside
  // spells resolved. An equality's endpoints are a leaf to this replacer, so a
  // projection standing in one is left untouched.
  ReadOnlyImplResolver reading(resolver);
  replacer.addReplacement([&reading](Type t) -> std::optional<Type> {
    auto proj = dyn_cast<ProjectionType>(t);
    if (!proj || isPolymorphicType(proj))
      return std::nullopt;
    auto resolved = reading.resolveProjectionType(proj);
    if (succeeded(resolved))
      return *resolved;
    return std::nullopt;
  });

  uint64_t positionsRespelled = 0;

  root->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    // Nothing writes into a template: its spelling is resolved when it is
    // cloned for a concrete instance, not by this sweep.
    if (isTemplate(op))
      return WalkResult::skip();

    DemandFrame frame(op->getLoc());

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
    if (movedHere && anchor)
      *anchor = op->getLoc();
    return WalkResult::advance();
  });

  // A sweep records no proof, so the count of facts does not move for it; what
  // it moves is the module's spelling of them, which is what proof derivation
  // reads. A sweep that respelled nothing leaves every derivation reading the
  // module it read.
  if (positionsRespelled != 0)
    resolver.noteRespelling(replacer);

  return positionsRespelled;
}

/// Proves a claim-producing op and replaces it with a trait.witness.
///
/// The proving obligation is keyed on the result ClaimType, not the
/// producing op: allege, derive, and project results all discharge through
/// this one rule. Which producers it matches follows from which constructor
/// built it: the fact-establishing use matches trait.allege alone, because a
/// claim derived inside a still-polymorphic body is not yet its business,
/// while the read-only driver use matches allege, derive and project results
/// alike.
///
/// Both registrations are permanent. A claim result the driver's own rewrites
/// produce is one no step before the driver could have seen, for the same reason
/// the projection resolution beside it is permanent: what the driver mints is
/// not what the module spelled when the round's commit swept it.
struct ProveClaimResultPattern : public RewritePattern {
  /// Impl selection itself, where this pattern may establish facts, and nothing
  /// where it may only read them.
  ImplResolver *minting;
  /// A read of what impl selection has settled, which every use has.
  ReadOnlyImplResolver reading;

  /// The step that establishes the facts the rest of the stage reads. It
  /// matches `trait.allege` alone: a claim derived inside a still-polymorphic
  /// body is not yet its business.
  ProveClaimResultPattern(MLIRContext *ctx, ImplResolver &resolver)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      minting(&resolver), reading(resolver) {}

  /// The instantiation driver, which reads what earlier steps established. A
  /// fact minted while the driver runs reaches nothing the driver's earlier
  /// rewrites saw, so a claim the record does not prove is declined and left
  /// for the step that can prove it.
  ProveClaimResultPattern(MLIRContext *ctx, const ReadOnlyImplResolver &reading)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      minting(nullptr), reading(reading) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter& rewriter) const override {
    if (minting ? !isa<AllegeOp>(op) : !isa<AllegeOp, DeriveOp, ProjectOp>(op))
      return failure();

    // In-place retyping by proof propagation can prove a claim out from under
    // its producing op, which leaves that op holding exactly the witness it was
    // going to build: its result type already names both the proof and the
    // application. Build it, rather than leaving behind a producer nothing
    // legalizes wherever a consumer still wants its result.
    auto claim = cast<ClaimType>(op->getResult(0).getType());

    // An equality claim is never proven by impl selection; a projection hop to a
    // trait's equality requirement is established by the requirement itself and
    // discharged when its endpoints ground-resolve at the leftover check.
    if (claim.isEquality())
      return rewriter.notifyMatchFailure(op, "equality claim is not impl-proved");

    if (claim.isProven()) {
      rewriter.replaceOpWithNewOp<WitnessOp>(op, claim.getProof(),
                                             claim.getTraitApplication());
      return success();
    }

    // skip polymorphic claims -- they can't be resolved until after monomorphization
    if (!claim.isMonomorphic())
      return rewriter.notifyMatchFailure(op, "polymorphic claim deferred");

    DemandFrame frame(op->getLoc());

    auto errFn = [&] { return op->emitOpError(); };

    // build or reuse canonical evidence for this claim
    FailureOr<FlatSymbolRefAttr> sym =
        minting ? minting->resolveAndEnsureProofFor(claim, rewriter, errFn)
                : reading.getRecordedProofFor(claim);
    if (failed(sym)) {
      if (!minting)
        (void)reading.decline(claim);
      return rewriter.notifyMatchFailure(op, "couldn't find proof of this claim");
    }

    // Mint the witness at the same spelling the proof was recorded under.
    // Impl selection resolves the claim's monomorphic projections before
    // recording (resolveImplFor), so the recorded fact is spelled with those
    // projections resolved. Spelling the witness at the producer's source claim
    // instead would leave the witnessed application and its recorded proof
    // disagreeing on the projections. Resolving here is deterministic recorded
    // lookup (the impls are already in the module) and is idempotent with the
    // resolution resolveAndEnsureProofFor just performed.
    auto recorded = cast<ClaimType>(
        minting ? minting->resolveProjectionsIn(claim, rewriter)
                : reading.resolveProjectionsIn(claim));
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

  // run convert-to-trait patterns
  if (failed(convertToTrait(module)))
    return failure();

  // verify traits are acyclic
  if (failed(verifyAcyclicTraits(module)))
    return failure();

  // verify that proofs named in declared signatures actually prove their claims
  if (failed(verifyDeclaredClaimProofs(module)))
    return failure();

  // an ImplResolver for this module
  ImplResolver resolver(module, ledger);

  MLIRContext *ctx = module.getContext();

  // apply rewrite patterns
  {
    RewritePatternSet patterns(ctx);
    patterns.add<ProveClaimResultPattern>(ctx, resolver);

    // rewrite trait.allege -> trait.witness. Shells are excluded: an allege
    // inside a polymorphic function is resolved when that function is cloned
    // for a concrete instance, so this driver never turns one into a witness.
    if (failed(applyPatternsOverReachableOps(module, std::move(patterns),
                                             GreedyRewriteConfig(),
                                             /*changed=*/nullptr,
                                             /*includeTemplateShells=*/false)))
      return failure();
  }

  // assert that no monomorphic trait.allege remain outside a template. A
  // template's allege is resolved when the template is cloned for a concrete
  // instance.
  bool hasLeftovers = false;
  module.walk([&](AllegeOp op) {
    if (!op.getClaim().isMonomorphic() || isForeign(op)) return;
    hasLeftovers = true;
    op.emitError() << "unresolved monomorphic trait.allege after resolve-impls";
  });
  if (hasLeftovers) return failure();

  // Normalize claim types: after allege→witness, a proof's type parameter
  // may itself contain a claim that was just proven.  Respell all
  // unproven claims in their proven forms so that downstream instantiation
  // sees consistent types.
  respellProvenClaimsInPlace(resolver, module);

  return resolver;
}

void ResolveImplsPass::runOnOperation() {
  if (failed(resolveImpls(getOperation())))
    signalPassFailure();
}


//===----------------------------------------------------------------------===//
// InstantiateMonomorphsPass
//===----------------------------------------------------------------------===//

/// Extend this substitution with bindings that resolve concrete `!trait.proj`
/// types visible after applying the current substitution.
void CallSubstitution::discoverProjectionBindings(
    TypeRange types, ModuleOp module, const ReadOnlyImplResolver &reading,
    bool &declined) {
  for (Type ty : types) {
    apply(ty).walk([&](Type t) {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj))
        return;
      if (projectionBindings.lookup(proj))
        return;
      auto resolved = reading.resolveProjectionType(proj);
      if (succeeded(resolved)) {
        projectionBindings.bind(proj, *resolved);
        return;
      }
      // The read answers from the impls selection has settled on, and it settles
      // one only for an application some round put to it. A projection exactly
      // one impl in the module binds is one selection would settle the same way
      // whenever it were asked, so it is read from the module here instead of
      // waited for: waiting costs a round and arrives at the same impl. Where
      // the module binds it with no impl or with several, the lookup declines
      // and records which, and the call closes over a projection it still
      // cannot spell concretely -- so it lowers in the round that serves it.
      Type byLookup = resolveProjectionsByLookup(
          Type(proj), module, DemandOrigin::CallSiteSpecialization,
          LookupScope::Ground);
      if (byLookup == Type(proj)) {
        declined = true;
        return;
      }
      projectionBindings.bind(proj, byLookup);
    });
  }
}

/// Read the proven-claim bindings visible after applying the current
/// substitution off the record, deriving only a pair the record has no answer
/// for.
LogicalResult CallSubstitution::readEvidenceBindings(
    TypeRange types, ModuleOp module, const ReadOnlyImplResolver &reading,
    llvm::function_ref<InFlightDiagnostic()> err) {
  for (Type ty : types) {
    Type rewritten = apply(ty);
    if (failed(bindProofsIn(rewritten, module, evidenceBindings,
                                     DemandOrigin::ProofRecording,
                                     &reading.getDerivationMemo(), err)))
      return failure();
  }
  return success();
}

FailureOr<CallSubstitution> CallSubstitution::forCall(
    SpecializationMap specialization, TypeRange operandTypes,
    TypeRange resultTypes, FunctionType formalTy, ModuleOp module,
    const ReadOnlyImplResolver &reading,
    llvm::function_ref<InFlightDiagnostic()> err) {
  CallSubstitution subst(std::move(specialization));

  // A projection over a claim argument is spelled one way while the argument is
  // an obligation and another once it names its proof, so the proofs this call
  // spells are read before any projection is put to the read.
  if (failed(subst.readEvidenceBindings(operandTypes, module, reading, err)))
    return failure();
  if (failed(subst.readEvidenceBindings(resultTypes, module, reading, err)))
    return failure();

  bool changed;
  bool declined;
  do {
    // The component maps grow monotonically; `bindingCount()` is the raw component sum
    // so it is not affected by fixed-point normalization of the merged map.
    size_t before = subst.bindingCount();

    // A projection the read could not answer this time round may be answered
    // by the bindings this iteration goes on to add, so only the last
    // iteration's declines say what this substitution is missing.
    declined = false;
    subst.discoverProjectionBindings(resultTypes, module, reading, declined);
    subst.discoverProjectionBindings(operandTypes, module, reading, declined);
    if (formalTy) {
      subst.discoverProjectionBindings(formalTy.getInputs(), module, reading,
                                       declined);
      subst.discoverProjectionBindings(formalTy.getResults(), module, reading,
                                       declined);
    }

    if (failed(subst.readEvidenceBindings(operandTypes, module, reading, err)))
      return failure();
    if (failed(subst.readEvidenceBindings(resultTypes, module, reading, err)))
      return failure();

    changed = subst.bindingCount() != before;
  } while (changed);

  // A substitution that cannot spell one of the call's projections would
  // specialize the callee against a spelling the projection still stands in,
  // and nothing afterwards revisits a callee already specialized. The demand is
  // recorded, so this call lowers in the round that serves it.
  if (declined)
    return failure();
  return subst;
}

/// Visits the sites of `root` at which instantiation can owe work: every
/// sub-type except the endpoints of an equality claim.
///
/// An equality claim's endpoints hold a proposition, not work. What stands in
/// one is a term the equation relates, discharged when the equality settles, so
/// a scan that judges obligations sees the equality claim itself and never what
/// it relates. Every such scan reads this walk, so the rule is stated once.
static void walkObligationSites(Type root,
                                llvm::function_ref<void(Type)> visit) {
  root.walk<WalkOrder::PreOrder>([&](Type sub) -> WalkResult {
    visit(sub);
    if (auto claim = dyn_cast<ClaimType>(sub))
      if (claim.isEquality())
        return WalkResult::skip();
    return WalkResult::advance();
  });
}

/// True when `root` carries an obligation instantiation has not yet discharged:
/// an unproven monomorphic application claim, or a ground projection (one whose
/// base is concrete and so resolves in place). These are exactly the demands the
/// two leftover checks refuse if one still stands on a result or block-argument
/// type at instantiate's exit, and, standing outside a template, what keeps
/// instantiation pending for the erase gate.
static bool typeCarriesStandingObligation(Type root) {
  bool found = false;
  walkObligationSites(root, [&](Type sub) {
    if (auto claim = dyn_cast<ClaimType>(sub)) {
      if (claim.isApplication() && claim.isMonomorphic() && !claim.isProven())
        found = true;
    } else if (auto proj = dyn_cast<ProjectionType>(sub)) {
      if (!isPolymorphicType(Type(proj)))
        found = true;
    }
  });
  return found;
}

namespace {

/// The common product of lowering either kind of trait call site: the callee
/// specialized for this call and the result types after applying the same
/// closed substitution.
struct SpecializedCallTarget {
  func::FuncOp callee;
  SmallVector<Type> resultTypes;
};

/// The template a call instantiates, named by the symbol its callee is reached
/// through: a free function's own symbol, a method's trait-qualified name. Two
/// instances of one template answer alike here whatever type arguments their
/// instance names were mangled from, which is what makes a chain of them
/// countable.
static Attribute instantiationTemplateKey(FuncCallOp op) {
  return op.getCalleeNameAttr();
}
static Attribute instantiationTemplateKey(MethodCallOp op) {
  return op.getMethodRefAttr();
}

/// Builds and closes the call-site substitution, uses it to specialize the
/// callee against `formalTy`, and computes the concrete result types for the
/// replacement call.
///
/// Refuses before cutting an instance whose chain already carries the depth
/// limit's worth of instances of the same template: a template that
/// instantiates itself at a larger type makes progress at every step, so the
/// chain is what says it never ends.
template <typename CallOpT>
static FailureOr<SpecializedCallTarget>
specializeCallTarget(CallOpT op, PatternRewriter &rewriter,
                     const ReadOnlyImplResolver &reading,
                     FunctionType formalTy) {
  ModuleOp module = op.getOperation()->template getParentOfType<ModuleOp>();

  Operation *caller =
      op.getOperation()->template getParentOfType<func::FuncOp>();
  Attribute templateKey = instantiationTemplateKey(op);
  InstantiationChain &chain = reading.getInstantiationChain();
  if (chain.depthAt(caller, templateKey) >= kInstantiationDepthLimit) {
    chain.noteLimitReached();
    InFlightDiagnostic diagnostic =
        op.emitOpError()
        << "reached the instantiation limit while instantiating "
        << templateKey << ": " << kInstantiationDepthLimit
        << " instances of it stand on the chain that reaches this call";
    nameChainEnds<std::pair<Operation *, Attribute>>(
        diagnostic, chain.chainTo(caller),
        [](InFlightDiagnostic &d, std::pair<Operation *, Attribute> frame) {
          d.attachNote(frame.first->getLoc())
              << "instantiated from " << frame.second;
        });
    return failure();
  }

  // Pass time: the comparison reads both signatures through the record of what
  // impl selection has settled, on top of the evidence the call itself carries.
  auto specialization = op.buildParameterSpecialization(&reading);
  if (failed(specialization)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't build substitution");
    return failure();
  }

  auto errFn = [&] { return op.emitOpError(); };
  auto subst = CallSubstitution::forCall(std::move(*specialization),
                                        op.getOperandTypes(),
                                        op.getResultTypes(), formalTy, module,
                                        reading, errFn);
  if (failed(subst))
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

  auto callee =
      op.getOrSpecializeCallee(rewriter, *subst, &reading.getDerivationMemo());
  if (failed(callee)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't get or specialize callee");
    return failure();
  }
  target.callee = *callee;
  chain.note(target.callee.getOperation(), caller, templateKey);
  return target;
}

struct FuncCallOpLowering : public OpRewritePattern<FuncCallOp> {
  ReadOnlyImplResolver reading;

  FuncCallOpLowering(MLIRContext *ctx, const ReadOnlyImplResolver &reading)
    : OpRewritePattern(ctx), reading(reading) {}

  LogicalResult matchAndRewrite(FuncCallOp callOp, PatternRewriter &rewriter) const override {
    // The one readiness law, checked before any demand is raised: monomorphic
    // operands, proven operand claims, a callee at module scope with a
    // signature.
    if (!isRewritableGenericCall(callOp))
      return rewriter.notifyMatchFailure(callOp, "not a rewritable generic call");

    // The predicate confirmed the callee's signature exists; read it again to
    // specialize against, as the substitution below reads it.
    auto formalTy = callOp.getCalleeFunctionType();
    if (failed(formalTy))
      return rewriter.notifyMatchFailure(callOp, "couldn't get callee function type");

    DemandFrame frame(callOp.getLoc());

    auto target = specializeCallTarget(callOp, rewriter, reading, *formalTy);
    if (failed(target))
      return failure();

    // Operands pass through untouched (as in MethodCallOpLowering). The
    // readiness law above established that every operand claim is proven, so
    // specialization never bakes an unprovable claim parameter into the callee:
    // an operand application claim reaches this point only after the impl backing
    // it resolved and its proof settled.
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
  ReadOnlyImplResolver reading;

  MethodCallOpLowering(MLIRContext *ctx, const ReadOnlyImplResolver &reading)
    : OpRewritePattern(ctx), reading(reading) {}

  LogicalResult matchAndRewrite(MethodCallOp op, PatternRewriter &rewriter) const override {
    // The one readiness law, checked before any demand is raised: monomorphic
    // operands, a proven receiver claim, proven argument claims, a method with a
    // signature.
    if (!isRewritableGenericCall(op))
      return rewriter.notifyMatchFailure(op, "not a rewritable generic call");

    // The predicate confirmed the method's signature exists; read it again to
    // specialize against, as the substitution below reads it.
    auto formalTy = op.getMethodFunctionType();
    if (failed(formalTy))
      return rewriter.notifyMatchFailure(op, "couldn't get method function type");

    DemandFrame frame(op.getLoc());

    auto target = specializeCallTarget(op, rewriter, reading, *formalTy);
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
///
/// This runs in the driver rather than in the commit that sweeps the module, and
/// that is where it belongs. A commit resolves what the module SPELLS; the
/// projections this meets are the ones a substitution MINTS while the driver is
/// running -- stamping a concrete argument into a projection spelling turns a
/// symbolic projection into a ground one that no earlier sweep could have seen.
/// Moving the work into the commit was built and measured: it cost 2-3% of a
/// compile and still left this pattern applying 91 times -- once per operation
/// it rewrote -- so it was refused and this is the resolution arm.
struct ResolveProjectionsPattern : public RewritePattern {
  ReadOnlyImplResolver reading;

  ResolveProjectionsPattern(MLIRContext *ctx, const ReadOnlyImplResolver &reading)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx), reading(reading) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // A template's projections are resolved when the template is cloned for a
    // concrete instance, not here; the worklist never collects a template's
    // interior, so this pattern sees only code carried to a target.
    if (!opMentionsType<ProjectionType>(op))
      return failure();

    DemandFrame frame(op->getLoc());

    AttrTypeReplacer replacer = makeGroundProjectionReplacer(
        [&](ProjectionType proj) -> std::optional<Type> {
      auto resolved = reading.resolveProjectionType(proj);
      if (failed(resolved)) {
        (void)reading.decline(proj);
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

/// Puts a claim a function's signature declares to impl selection, which the
/// freeze standing over the instantiation driver forbids.
///
/// The driver's own patterns read the facts the steps before them recorded and
/// put nothing to selection, so nothing the compiler builds reaches the freeze.
/// This is what exercises it. A round collects the claims a result type or a
/// block argument spells, so a claim on a function carrying a body is one a
/// round has already collected; what is left for this pattern is a claim living
/// in a function type alone, which is a declaration with no body to spell it.
/// Selection meets that application for the first time here, finds no
/// candidate, and asks the generators -- which is the ask the freeze turns into
/// a fatal naming the claim and the span. Only the dialect's plugin adds this
/// pattern; the passes the compiler creates never do.
struct AskImplSelectionForADeclaredClaimPattern
    : public OpRewritePattern<func::FuncOp> {
  ImplResolver &resolver;

  AskImplSelectionForADeclaredClaimPattern(MLIRContext *ctx,
                                           ImplResolver &resolver)
      : OpRewritePattern(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    for (Type input : op.getFunctionType().getInputs()) {
      auto claim = dyn_cast<ClaimType>(input);
      if (!claim || claim.isProven() || !claim.isMonomorphic())
        continue;
      (void)resolver.resolveAndEnsureProofFor(claim, rewriter);
    }
    // Asking is all this does, so it rewrites nothing and the driver moves on.
    return failure();
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
  /// Demands impl selection resolved.
  uint64_t served = 0;
  /// Ops impl selection inserted serving them.
  uint64_t insertedServingDemands = 0;
  /// Type positions the round's commit respelled.
  uint64_t respelled = 0;
  /// Whether the instantiation driver rewrote anything.
  bool instantiated = false;
  /// Facts impl selection minted while the instantiation driver ran. A fact
  /// minted there reaches nothing the driver's earlier rewrites saw, so the
  /// round's own work is what this counts, and it counts writes rather than
  /// entries: an optimistic proof a failed recursion takes back out still
  /// moved the fact base the rewrites before it read.
  uint64_t instantiateMinted = 0;
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

/// The demands a round puts to impl selection: what `module` spells, and what a
/// recording engine declined and `ledger` therefore holds.
///
/// The two reach one population two ways, and neither reaches all of it. The
/// module is where a demand a later round can still serve must be standing, so
/// walking it finds every such demand wherever it was raised -- including the
/// ones nothing declined, because the step that would have declined them never
/// ran. What the module does not spell is what a component minted while it was
/// working and did not write down: a spelling a substitution built, or one a
/// candidate probe reached. Those exist only in what the engine that met them
/// recorded.
///
/// A demand leaves the drain for good when nothing a later round could ask
/// would settle it differently: impl selection resolved it, or refused it on
/// the arm no later resolution overturns. `drained` holds those. One selection
/// could not serve yet stays on the drain, and `attempted` carries the fact
/// epoch it was last put to selection at, so a round asks about it again
/// exactly where selection has minted something since -- which is the only
/// thing that can make the answer differ, and the only thing that keeps asking
/// again from asking the same question forever. Both halves of the union are
/// held to that same discipline, so a key the walk keeps finding is asked about
/// exactly as often as one an engine recorded once.
///
/// The order is the ledger's first, then the walk's, and both are the order
/// their own source produced: one run's rounds ask in the order another run's
/// do.
static SmallVector<Type>
collectUndrainedDemands(ModuleOp module, const DemandLedger &ledger,
                        const DenseSet<Type> &drained,
                        const DenseMap<Type, uint64_t> &attempted,
                        uint64_t epoch, DenseMap<Type, Location> &origins) {
  SmallVector<Type> collected;
  DenseSet<Type> taken;
  auto take = [&](Type demand) {
    if (drained.contains(demand))
      return;
    auto it = attempted.find(demand);
    if (it != attempted.end() && it->second == epoch)
      return;
    if (!taken.insert(demand).second)
      return;
    collected.push_back(demand);
  };
  for (Type demand : ledger.getDrainableDemands())
    take(demand);
  for (Type demand : demandsSpelledIn(module, /*inAttributes=*/true,
                                      DemandSkip::Infrastructure,
                                      DemandSkip::Infrastructure, &origins))
    take(demand);
  return collected;
}

/// Puts every demand in `collected` to impl selection, which generates the impl
/// the demand needs when none binds its application and partitions the
/// candidates when several do, and records what each attempt settled.
///
/// A demand selection resolved or refused for good leaves the drain; one it
/// could not serve yet stays, against the epoch it was asked at.
static void serveCollectedDemands(ImplResolver &resolver,
                                  ArrayRef<Type> collected,
                                  const DenseMap<Type, Location> &origins,
                                  OpBuilder &builder,
                                  DenseSet<Type> &drained,
                                  DenseSet<Type> &served,
                                  DenseMap<Type, uint64_t> &attempted,
                                  RoundWork &work) {
  for (Type demand : collected) {
    // A demand found by walking the module is named while it is put to
    // selection, so what the ask raises underneath is attributed to the op
    // carrying the spelling. A demand an engine recorded already carries where
    // it was raised, and the frame it was raised under is gone by now.
    std::optional<DemandFrame> spelledAt;
    if (auto origin = origins.find(demand); origin != origins.end())
      spelledAt.emplace(origin->second);
    // The epoch is read per demand rather than once per round: serving one
    // demand mints facts the demands after it in this batch are resolved
    // under, so a demand asked about before that is one the next round asks
    // about again.
    attempted[demand] = resolver.getFactEpoch();

    // Every engine whose declining leaves a demand standing declines a
    // monomorphic projection or an unproven monomorphic claim, so this is
    // total over what the drain holds.
    ImplResolver::DemandDisposition disposition;
    if (auto projection = dyn_cast<ProjectionType>(demand))
      disposition = resolver.serveDemand(projection, builder);
    else if (auto claim = dyn_cast<ClaimType>(demand))
      disposition = resolver.serveDemand(claim, builder);
    else
      llvm_unreachable("a drainable demand is a projection or a claim an "
                       "engine left spelled");

    switch (disposition) {
    case ImplResolver::DemandDisposition::Served:
      ++work.served;
      drained.insert(demand);
      served.insert(demand);
      break;
    case ImplResolver::DemandDisposition::Refused:
      drained.insert(demand);
      break;
    case ImplResolver::DemandDisposition::Deferred:
      break;
    }
  }
}

/// Checks that impl selection left nothing part-way done.
///
/// Selection is entered at round zero and at the round's own serving step,
/// where the drain puts demands to it; the instantiation driver only reads what
/// those steps settled. A reader of its facts between any two of those points
/// must find every application it opened closed, because a fact read part-way
/// through is one that is not yet a fact.
static void checkResolutionBoundary(const ImplResolver &resolver) {
  assert(resolver.isQuiescent() &&
         "impl selection must not be part-way through an application at a "
         "boundary between the stage's steps");
}

/// The resolvers, module-body builder, and module every projection-settlement
/// helper reads but never varies as it descends. Recorded facts and
/// obligation-holding selection come from `reading` and `resolver`; the impls a
/// resolution chain drives selection to generate insert at the module body
/// through `proofBuilder`; `module` anchors the shared fixed-point normalization
/// every resolution walk here runs, which owns the bound that cuts a chain that
/// never grounds out.
struct ProjectionSettleContext {
  const ReadOnlyImplResolver &reading;
  ImplResolver &resolver;
  OpBuilder &proofBuilder;
  ModuleOp module;
};

/// Resolve one hop of a monomorphic projection through an obligation-holding
/// impl. The recorded facts answer first; a projection nothing recorded is put
/// to impl selection, which resolves it only through an impl whose obligations
/// hold and refuses it otherwise. Selection's own sub-resolutions are
/// discardable probes of the module -- not demands this stage undertook to
/// serve -- so they are marked speculative and never reach the drain. Nothing
/// where the projection has no obligation-holding impl.
static std::optional<Type>
resolveProjectionHop(ProjectionType proj, const ProjectionSettleContext &settle) {
  if (auto recorded = settle.reading.resolveProjectionType(proj);
      succeeded(recorded))
    return *recorded;
  SpeculationScope speculation;
  if (auto selected =
          settle.resolver.resolveProjectionType(proj, settle.proofBuilder);
      succeeded(selected))
    return *selected;
  return std::nullopt;
}

/// Resolve to a fixed point every ground projection standing anywhere in `type`,
/// descending composites, through `hop`. A projection `hop` declines and any
/// polymorphic projection are left standing. Resolution runs to a fixed point
/// because one hop's binding may spell the next; the shared normalizer owns the
/// bound and stops the compilation at a chain that never grounds out, the same
/// refusal every ground resolver makes.
static Type resolveGroundProjections(
    Type type, ModuleOp module,
    llvm::function_ref<std::optional<Type>(ProjectionType)> hop) {
  return normalizeProjectionsToFixedPoint(type, module, [&](Type current) {
    AttrTypeReplacer replacer = makeGroundProjectionReplacer(hop);
    return replacer.replace(current);
  });
}

/// Whether a monomorphic equality claim is settled at the leftover check: its
/// two endpoints ground-resolve to one spelling through impls whose obligations
/// hold.
///
/// An equality claim carries no proof -- its evidence is the value
/// itself -- so unlike an application claim it is never "proven"; it is
/// discharged instead when the projections in its endpoints resolve and the two
/// endpoints meet at one ground type. Each projection resolves first from what
/// impl selection has recorded, the spelling every projection some round put to
/// selection already carries. The rounds hold an equality's endpoints as a leaf
/// and never put the projections inside them to selection, so a
/// projection whose resolution chain runs through an impl nothing else demanded
/// has no recorded outcome; such a projection is put to selection here instead.
/// Selection resolves a projection only through an impl whose assumptions are
/// satisfiable and refuses it otherwise, so a projection whose only candidate
/// impl is conditional with an undischarged assumption stays spelled and the
/// endpoints do not meet -- the settlement never resolves through an impl whose
/// where-bounds do not hold. Resolution runs to a fixed point because one hop's
/// binding may spell the next. The endpoints are read through the accessor and
/// rebuilt atomically, the one sanctioned way to move one.
static bool equalityClaimGroundResolvesToOneSpelling(
    ClaimType claim, const ProjectionSettleContext &settle) {
  auto eq = claim.getEqualityAttr();
  if (!eq)
    return false;
  // Resolve each endpoint's ground projections to a fixed point through the
  // shared normalizer (recorded facts first, then impl selection -- see the
  // helpers).
  auto resolveEndpoint = [&](Type endpoint) -> Type {
    return resolveGroundProjections(endpoint, settle.module,
                                    [&](ProjectionType proj) {
      return resolveProjectionHop(proj, settle);
    });
  };
  Type lhs = resolveEndpoint(eq.getLhs());
  Type rhs = resolveEndpoint(eq.getRhs());
  return lhs == rhs && isGroundType(lhs);
}

/// The settlement resolvers, per-site builder, and location a projection-
/// resolution witness mint reads but never varies as the chain walks hop to
/// hop. Recorded facts, obligation-holding selection, and the module-body
/// builder impl generation goes through come from `settle`; the witness and its
/// premise witnesses insert at the consumer through `witnessBuilder`, so they
/// dominate it, while premise proofs and any impl generation go to the module
/// body through `settle.proofBuilder`; every op carries `loc`.
struct ProjectionResolveMintContext {
  ProjectionSettleContext settle;
  Location loc;
  OpBuilder &witnessBuilder;
};

/// Mint the proj-resolve witness proving `<proj = binding>`, where the impl
/// `proj`'s trait application resolves through binds the projected member to
/// `binding` in one hop, paired with that binding. Resolution goes through the
/// same obligation-holding selection the settlement ran, so the cited impl is
/// one whose bounds hold. Premises discharging that impl's own assumptions ride
/// along, so a witness citing a conditional impl passes obligation-discharge
/// verification. The minting `ctx` supplies the resolvers and builders. Fails
/// where the projection has no obligation-holding impl.
static FailureOr<std::pair<Value, Type>>
mintProjectionResolutionWitness(ProjectionType proj,
                                const ProjectionResolveMintContext &ctx) {
  MLIRContext *mlirCtx = proj.getContext();
  auto binding = resolveProjectionHop(proj, ctx.settle);
  if (!binding)
    return failure();
  ClaimType selfClaim = ClaimType::get(mlirCtx, proj.getTraitApplication());
  auto resolvedImpl = ctx.settle.reading.getRecordedImplFor(selfClaim);
  if (failed(resolvedImpl))
    return failure();
  SmallVector<Value> obligationPremises;
  if (!resolvedImpl->impl.isUnconditional()) {
    auto assumptions = resolvedImpl->impl.specializeAssumptionsAsClaimsFor(
        resolvedImpl->selectedClaim);
    if (failed(assumptions))
      return failure();
    for (ClaimType assumption : *assumptions) {
      auto proof = ctx.settle.resolver.resolveAndEnsureProofFor(
          assumption, ctx.settle.proofBuilder);
      if (failed(proof))
        return failure();
      obligationPremises.push_back(
          WitnessOp::create(ctx.witnessBuilder, ctx.loc, *proof,
                            assumption.getTraitApplication())
              .getResult());
    }
  }
  TypeEqualityAttr equality =
      TypeEqualityAttr::get(mlirCtx, Type(proj), *binding);
  auto witness_attr = WitnessAttr::get(
      mlirCtx, Attribute(equality),
      FlatSymbolRefAttr::get(mlirCtx, resolvedImpl->impl.getSymName()));
  Value witness = WitnessOp::create(ctx.witnessBuilder, ctx.loc, equality, witness_attr,
                                    obligationPremises)
                      .getResult();
  return std::make_pair(witness, *binding);
}

/// Walk a projection endpoint to its ground spelling, appending one witness for
/// every ground projection resolved along the way. The walk descends composites
/// exactly as the settlement decision does, so a projection nested inside one --
/// the resolved side of `type Out = Vec<Self::Item>`, where the projection sits
/// inside the vector -- yields its evidence just as a top-level projection does;
/// resolution runs to a fixed point because a resolved binding may itself spell
/// a projection. A concrete endpoint contributes no witness. A projection with
/// no obligation-holding impl fails; a chain that never grounds out stops the
/// compilation at the shared normalizer, the same refusal every ground resolver
/// makes.
static LogicalResult
mintProjectionResolveChain(Type endpoint,
                           const ProjectionResolveMintContext &ctx,
                           SmallVector<Value> &witnesses) {
  // A hop that cannot mint its witness fails the whole chain, but the shared
  // resolver only leaves such a projection standing; this side channel carries
  // that failure out past the fixed point so the outer result short-circuits
  // rather than falling through to the unresolved-projection check below.
  LogicalResult mintOutcome = success();
  Type current = resolveGroundProjections(
      endpoint, ctx.settle.module, [&](ProjectionType proj) -> std::optional<Type> {
        auto witness = mintProjectionResolutionWitness(proj, ctx);
        if (failed(witness)) {
          mintOutcome = failure();
          return std::nullopt;
        }
        witnesses.push_back(witness->first);
        return witness->second;
      });
  if (failed(mintOutcome))
    return failure();
  // A ground projection still standing at the fixed point never resolved.
  bool unresolved = false;
  current.walk([&](Type sub) {
    if (auto proj = dyn_cast<ProjectionType>(sub))
      if (!isPolymorphicType(proj))
        unresolved = true;
  });
  return failure(unresolved);
}

/// Replace a ground-resolvable equality `trait.assume` with the witness that
/// proves its equality, inserted at the assume so it dominates the assume's
/// uses.
///
/// An equality assume is an axiom the enclosing scope inherited (an impl's or
/// trait's where-clause equality re-established inside a method body). Where it
/// feeds only a `trait.coerce` that folds once the projection resolves, the
/// assume goes dead and is eliminated; where it feeds a use that keeps the
/// equality claim -- an operand of an already-ground callee that retains the
/// parameter -- nothing consumes it, and a bare `trait.assume` is an axiom no
/// legalization removes. Now that the equality ground-resolves, this mints the
/// evidence that proves it, exactly as codegen mints where the equality is
/// first established: a refl marker for identical endpoints, or one proj-resolve
/// witness per hop of each projection an endpoint carries -- nested in a
/// composite or standing alone, since a resolved binding may itself spell a
/// projection -- each citing the impl that
/// binds one hop, with application-arm premises discharging a conditional impl's
/// own assumptions so the witness passes verification. A lone witness
/// that already proves the result equality outright is that witness when the
/// orientation matches; otherwise the hops' witnesses compose to the result
/// equality, whose ground congruence closure carries the endpoints together
/// across every hop and is direction-blind. Settlement gates this reduction on
/// the same fixed-point resolution reaching one ground spelling within the hop
/// bound, so the walk here terminates; a chain that never grounds out stops the
/// compilation at the shared normalizer. Proofs the premises need are minted
/// through `settle.proofBuilder` at the module body; the witnesses themselves
/// are inserted at the assume.
static LogicalResult reduceGroundEqualityAssume(
    AssumeOp assume, TypeEqualityAttr eq, const ProjectionSettleContext &settle) {
  MLIRContext *ctx = assume.getContext();
  Location loc = assume.getLoc();
  Type lhs = eq.getLhs();
  Type rhs = eq.getRhs();

  OpBuilder builder(ctx);
  builder.setInsertionPoint(assume);

  auto replaceWith = [&](Value witness) {
    assume.getResult().replaceAllUsesWith(witness);
    assume.erase();
  };

  // refl: identical endpoints carry their own evidence.
  if (lhs == rhs) {
    replaceWith(WitnessOp::create(builder, loc, eq).getResult());
    return success();
  }

  // Mint each endpoint's per-hop resolution chain (see the function doc);
  // no premises means the endpoints did not ground-resolve.
  ProjectionResolveMintContext mintCtx{settle, loc, builder};
  SmallVector<Value> premises;
  if (failed(mintProjectionResolveChain(lhs, mintCtx, premises)) ||
      failed(mintProjectionResolveChain(rhs, mintCtx, premises)))
    return failure();
  if (premises.empty())
    return failure();

  // A sole hop witness that already proves the result equality is used as is;
  // otherwise the hops' witnesses compose to it below.
  if (premises.size() == 1)
    if (auto sole = cast<WitnessOp>(premises.front().getDefiningOp());
        sole.getResultClaim().getEqualityAttr() == eq) {
      replaceWith(premises.front());
      return success();
    }
  replaceWith(
      WitnessOp::create(builder, loc, eq, ValueRange(premises)).getResult());
  return success();
}


/// Every function `module` holds mentions only the type parameters its own
/// declaration binds.
///
/// The walk reaches every function, template or not, called or not: a template
/// is exactly what the stage carries to no target, so nothing downstream reads
/// its interior, and a stray parameter inside one rides into every clone made
/// from it. Each function reports on its own, so one run names them all.
LogicalResult verifyFunctionBodiesAreWellScoped(ModuleOp module) {
  bool wellScoped = true;
  module.walk([&](func::FuncOp function) {
    if (failed(verifyFunctionBodyIsWellScoped(function)))
      wellScoped = false;
  });
  return success(wellScoped);
}

} // end namespace

/// `askImplSelectionForImpls` adds the pattern that puts a declared claim to
/// impl selection from inside the instantiation driver, which is how the freeze
/// over that driver is exercised. Only the dialect's plugin passes it.
LogicalResult instantiateMonomorphs(ModuleOp module,
                                    bool askImplSelectionForImpls) {
  // Well-scopedness leads, before anything is read, folded, collected or
  // cloned: a body that mentions a type parameter its declaration does not bind
  // has no source for that parameter, and every step after this one either skips
  // the body (a template's interior) or rewrites it, at which point the stray
  // parameter has become a clone's or has folded away unseen.
  if (failed(verifyFunctionBodiesAreWellScoped(module)))
    return failure();

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

  // The resolver was moved out of the sub-phase that built it, so its ledger is
  // reinstalled here to span this sub-phase's rounds and leftover walks.
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
  // same stack that raised them and never reach the drain. The projections and
  // claims the instantiation driver meets it reads rather than resolves,
  // declining any the recorded facts do not yet answer; each decline is a demand
  // a later round collects and serves. Anything still standing at the end of the
  // stage is pinned by the leftover walks.
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
  // What the record stood at when the commit last swept. The sweep reads the
  // record and rewrites the module through it, so a record that has gained no
  // answer since leaves the sweep with nothing the last one did not already do.
  uint64_t recordAtSweep = resolver->getRecordEpoch();
  // Whether the module stands where the instantiation driver last left it, at
  // that driver's own fixed point. A driver run that minted nothing leaves it
  // there; a run that minted is a run whose earlier rewrites read facts its
  // later ones did not, so what it left is not a fixed point under the facts
  // as they now stand. No round has run the driver yet, so the first one runs
  // it unconditionally.
  bool atInstantiationFixedPoint = false;
  // Where the last commit moved something, so that the round bound's refusal
  // names somewhere the stage's work landed.
  std::optional<Location> lastRespelled;
  for (bool wrote = true; wrote;) {
    if (++round > maxRounds) {
      InFlightDiagnostic diagnostic =
          emitError(lastRespelled.value_or(module.getLoc()));
      return diagnostic
             << "instantiate-monomorphs did not converge: the stage ran its "
                "rounds to the round bound, which indicates a round writing "
                "work back for the next one to find";
    }

    RoundWork work;
    uint64_t epochAtRoundHead = resolver->getFactEpoch();
    uint64_t recordAtRoundHead = resolver->getRecordEpoch();

    // FLUSH. Every refusal a later resolution could answer differently is
    // forgotten here, so that the questions the rest of the round asks are
    // asked against the facts as they now stand.
    resolver->forgetRetriableRefusals();

    // BRIDGE. The patterns that lift another dialect's vocabulary into trait
    // claims run whenever something has written to the module since they last
    // ran, because that writing may have created the ops they lift.
    if (writtenSinceBridge) {
      if (failed(convertToTrait(module, &work.bridged)))
        return failure();
      writtenSinceBridge = false;
      writtenSinceSweep |= work.bridged;
    }

    // COLLECT.
    DenseMap<Type, Location> spelledAt;
    SmallVector<Type> collected = collectUndrainedDemands(
        module, resolver->getDemandLedger(), drained, attempted,
        resolver->getFactEpoch(), spelledAt);
    size_t drainAtCollect =
        resolver->getDemandLedger().getDrainableDemands().size();

    // GENERATE.
    {
      RoundInsertionCounts insertions;
      OpBuilder builder(ctx);
      builder.setListener(&insertions);
      builder.setInsertionPointToEnd(module.getBody());
      serveCollectedDemands(*resolver, collected, spelledAt, builder, drained,
                            served,
                            attempted, work);
      work.insertedServingDemands = insertions.inserted;
    }
    writtenSinceBridge |= work.insertedServingDemands != 0;
    writtenSinceSweep |= work.insertedServingDemands != 0;

    // COMMIT. Every claim the stage has proved is respelled in its proven form
    // throughout the module, so the round that follows reads one spelling of
    // each claim wherever it appears.
    //
    // The sweep rewrites where the module spells something the record answers
    // for, so a module nothing has written to since the last sweep, under a
    // record that has gained no answer since, has nothing left for it to move.
    // The record rather than the count of proofs, because settling an
    // application whose impl the module already held mints no proof and still
    // gives the sweep an answer the last one did not have.
    if (writtenSinceSweep || resolver->getRecordEpoch() != recordAtSweep) {
      work.respelled =
          respellProvenClaimsInPlace(*resolver, module, &lastRespelled);
      // Sampled after the sweep, which moves the record itself wherever it
      // respelled what a proof is read through.
      recordAtSweep = resolver->getRecordEpoch();
      writtenSinceSweep = false;
      writtenSinceBridge |= work.respelled != 0;
    }

    checkResolutionBoundary(*resolver);

    // INSTANTIATE. Rewrite trait.func.call and trait.method.call, prove claim
    // producers (allege, derive, project), resolve projections, and monomorphize
    // any generic op whose results become monomorphic.
    // The driver reads what the steps before it established. Serving a demand
    // is the round's own work, done where a round can see what it minted.
    //
    // Nothing the steps above did moved what the driver reads when the bridge
    // lifted nothing and the record gained no answer. The driver's own patterns
    // serve from the record, so one quantity stands for all of what the steps
    // above could have given them: serving a demand settles an application there
    // whether or not it mints, and the commit's respelling moves it too. The
    // bridge is named beside it because it rewrites the module without the
    // resolver hearing of it. Under both the driver would be handed the module
    // its own last run left at that run's fixed point, together with the record
    // that run read, so it would apply no pattern: the round skips it,
    // `instantiated` stays false, and the loop ends unless something else this
    // round wrote.
    //
    // Neither the flush nor a refusal is among them. A read fails on a refused
    // application exactly as it fails on one selection has never been asked
    // about, so neither writing the refusal nor dropping it again moves the
    // record; a round whose only work was to refuse an application, or to forget
    // that it had, hands the driver exactly what its last run left.
    //
    // What the driver reads beyond its own rewrites is the facts the module
    // spells -- trait declarations, impl headers and their associated-type
    // bindings, and proof ops -- and those move only where the steps above move
    // them. No pattern written here rewrites any of them, and neither extension
    // point that accepts a foreign pattern may contribute one that does:
    // populateInstantiateMonomorphsPatterns below, and
    // populateConvertToTraitPatterns in the bridge above.
    bool instantiationInputMoved =
        work.bridged || resolver->getRecordEpoch() != recordAtRoundHead;
    if (!atInstantiationFixedPoint || instantiationInputMoved) {
      ReadOnlyImplResolver reading(*resolver);
      RewritePatternSet patterns(ctx);
      patterns.add<ProveClaimResultPattern>(ctx, reading);
      patterns.add<MonomorphizeResultTypesPattern>(ctx);
      patterns.add<FuncCallOpLowering>(ctx, reading);
      patterns.add<MethodCallOpLowering>(ctx, reading);
      patterns.add<ResolveProjectionsPattern>(ctx, reading);
      if (askImplSelectionForImpls)
        patterns.add<AskImplSelectionForADeclaredClaimPattern>(ctx, *resolver);

      // collect instantiate-monomorphs patterns from other dialects
      for (Dialect *d : ctx->getLoadedDialects()) {
        if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
          iface->populateInstantiateMonomorphsPatterns(patterns);
      }

      GreedyRewriteConfig config;
      config.setMaxNumRewrites(rewriteBudgetFor(module));

      uint64_t epochAtInstantiate = resolver->getFactEpoch();
      {
        // Generating an impl is a round's own work, and one generated while the
        // driver runs is a fact the run's earlier rewrites could not see. The
        // driver's patterns read what the steps before them recorded and put
        // nothing to impl selection, so nothing under this reaches the generator
        // arm; the freeze is what says so.
        ImplGenerationFreeze freeze(*resolver, "the instantiation driver");
        LogicalResult instantiated = applyPatternsOverReachableOps(
            module, std::move(patterns), config, &work.instantiated,
            /*includeTemplateShells=*/false);
        work.instantiateMinted = resolver->getFactEpoch() - epochAtInstantiate;
        // A generation ask under the freeze already emitted its report; fail the
        // stage rather than converge over the broken contract (the greedy driver
        // treats the ask's failure as a pattern that did not apply).
        if (freeze.wasAsked())
          return failure();
        if (failed(instantiated))
          return module.emitError(
              "instantiate-monomorphs did not converge: rewrite budget exceeded, "
              "which indicates a non-confluent pattern pair cycling on a type "
              "spelling");
      }
      atInstantiationFixedPoint = work.instantiateMinted == 0;
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
  }

  // A call refused on the instantiation depth limit has already reported
  // itself, and the greedy driver took that refusal for a pattern that did not
  // apply. The stage fails here rather than converging over a chain it stopped.
  if (resolver->getInstantiationChain().wasLimitReached())
    return failure();

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
  ReadOnlyImplResolver reading(*resolver);
  // Settling an equality claim resolves the projections in its endpoints, which
  // may put an undemanded impl to selection and generate the impl a resolution
  // chain runs through. That inserts into the module, so the candidate claims
  // are gathered under the walk and judged after it closes, never while the
  // walk holds the module open.
  SmallVector<std::pair<Operation *, ClaimType>> monomorphicClaims;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op))
      return WalkResult::skip();
    for (Value result : op->getResults()) {
      walkObligationSites(result.getType(), [&](Type sub) {
        auto claim = dyn_cast<ClaimType>(sub);
        if (claim && !claim.isProven() && claim.isMonomorphic())
          monomorphicClaims.emplace_back(op, claim);
      });
    }
    return WalkResult::advance();
  });
  // Settling an equality may reach impl generation -- for a resolution chain no
  // round demanded, or a satisfiability probe of a conditional impl -- and impl
  // generation requires a builder whose insertions are observed, so this builder
  // must carry a listener. Its presence is the precondition; its tally is not
  // read here, because unlike the round loop above (where the insertion count
  // decides whether another round runs) no round follows this leftover check. A
  // generated impl is a complete monomorphic definition the module verifier
  // checks, inserted at the module body where it belongs.
  RoundInsertionCounts settleInsertions;
  OpBuilder settleBuilder(ctx);
  settleBuilder.setListener(&settleInsertions);
  settleBuilder.setInsertionPointToEnd(module.getBody());
  ProjectionSettleContext settle{reading, *resolver, settleBuilder, module};
  for (auto [op, claim] : monomorphicClaims) {
    // An equality claim has no proof to await; it is settled when its endpoints
    // ground-resolve to one spelling through impls whose obligations hold. A
    // monomorphic equality that resolves is not a leftover; one whose projection
    // has no obligation-holding impl stays unequal and is reported like an
    // unprovable application claim.
    if (claim.isEquality() &&
        equalityClaimGroundResolvesToOneSpelling(claim, settle)) {
      // A surviving equality `trait.assume` is an inherited axiom no
      // legalization removes; now that it ground-resolves, replace it with the
      // witness proving it. Producers already carrying legal evidence (a
      // `trait.witness`) need nothing here.
      if (auto assume = dyn_cast<AssumeOp>(op))
        if (failed(reduceGroundEqualityAssume(assume, claim.getEqualityAttr(),
                                              settle))) {
          hasLeftovers = true;
          op->emitError()
              << "ground-resolvable equality assumption " << claim
              << " could not be reduced to a witness after "
                 "instantiate-monomorphs";
        }
      continue;
    }
    hasLeftovers = true;
    op->emitError() << "unproven monomorphic claim " << claim
                    << " after instantiate-monomorphs";
  }
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
  // and its whole subtree is skipped. The scan reads the obligation sites of a
  // type, so a projection standing in an equality's endpoints is the equality
  // settling's to discharge and is not reported here.
  bool sawUnresolvedProjection = false;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op))
      return WalkResult::skip();
    auto report = [&](Type root) {
      walkObligationSites(root, [&](Type sub) {
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

  // A generic call the rounds could still rewrite but did not is a call whose
  // callee specialization fail-closed -- an external polymorphic declaration has
  // no body to clone. Standing outside every template, it is named here rather
  // than left for a later step to meet a call it cannot lower.
  bool sawSurvivingCall = false;
  module.walk([&](Operation *op) {
    if (isForeign(op) || !isRewritableGenericCall(op))
      return;
    op->emitOpError()
        << "rewritable generic call survived instantiate-monomorphs";
    sawSurvivingCall = true;
  });
  if (sawSurvivingCall)
    return failure();

  // The two walks above reject a demand still spelled on an op result or block
  // argument. A demand can also stand at a place they do not reach -- a claim on
  // a block argument the leftover-claim walk passes over, or a projection stored
  // in an attribute neither walk reads -- and a demand deferred to a round that
  // never came stands with no spelling change to find. This backstops both: a
  // drainable demand the stage never served that is still spelled anywhere is a
  // demand the stage undertook to serve and did not.
  if (failed(resolver->getDemandLedger().checkStandingDemandsServed(module,
                                                                    served)))
    return failure();

  // ModuleOp's own verifier hook runs here over the module shell -- it does not
  // recurse into the body, which is not guaranteed to verify recursively in
  // general: a marked coerce whose two projections grounded to different types
  // can stand here, and this shallow tail does not judge it. The bonded erase
  // pass refuses such a coerce at its barrier, where endpoints that stand apart
  // cannot be discharged and cannot cross.
  DemandRecordingSuspension verifying;
  return module.verify();
}

void InstantiateMonomorphsPass::runOnOperation() {
  if (failed(instantiateMonomorphs(getOperation(),
                                   /*askImplSelectionForImpls=*/false)))
    signalPassFailure();
}

std::unique_ptr<Pass> createInstantiateMonomorphsPass() {
  return std::make_unique<InstantiateMonomorphsPass>();
}

void AskImplSelectionDuringInstantiationPass::runOnOperation() {
  if (failed(instantiateMonomorphs(getOperation(),
                                   /*askImplSelectionForImpls=*/true)))
    signalPassFailure();
}


//===----------------------------------------------------------------------===//
// ErasePolymorphsPass
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

struct EraseCoerceOp : public OpConversionPattern<CoerceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(CoerceOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Erasure is a checked judgment on the op, not a trusting forward. Every
    // cited equality is a claim value that maps to zero here, and so is a
    // claim-typed input (a claim-to-claim coerce shrinking 1:0). A coerce whose
    // input itself erased, or whose result is unused, leaves nothing to forward
    // -- but dropping it is still conditioned on its own recorded endpoints: a
    // discharged respell carries endpoints that coincide here (projection
    // resolution rewrote both to the same ground type), while an undischarged one
    // (its two projections ground to different types) still relates two spellings
    // and may not cross the barrier unjudged: every claim-to-claim coerce that
    // reaches the barrier live, proven or marked, is judged here (one dropped as
    // dead earlier, during instantiate-monomorphs, forwarded no value and so needs
    // no judgment). Comparison strips application-claim proofs, exactly as the verifier does:
    // a coerce compares modulo the proof permanently, so exchanging a proof
    // label alone is not a surviving difference. (The value-carrying arm below
    // needs no strip; the values it forwards carry no proof.)
    ValueRange input = adaptor.getInput();
    if (input.empty() || op.getResult().use_empty()) {
      if (stripClaimProofs(op.getInput().getType()) !=
          stripClaimProofs(op.getResult().getType()))
        return rewriter.notifyMatchFailure(
            op, "a coerce whose endpoints still differ after conversion is not "
                "discharged and cannot cross the erase barrier");
      rewriter.eraseOp(op);
      return success();
    }
    // The input survived as one value. Forwarding it is a no-op exactly when its
    // post-conversion type equals the result type -- the discharged (reflexive)
    // form, which projection resolution has produced by here. An undischarged
    // coerce still relates two different types; it is refused, so the op stays
    // illegal and the conversion fails loudly. The witness is deliberately
    // not re-verified: the replay endpoints are authoritative at the barrier.
    if (input.front().getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, input);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "a coerce whose endpoints still differ after conversion is not "
            "discharged and cannot cross the erase barrier");
  }
};


/// The first type of the trait dialect reachable in `root` that `admitted` does
/// not accept, or a null type when every one it finds is admitted. `root` is a
/// type or an attribute: one walk reads either, since a sub-element walk reaches
/// every type held below the root whichever kind the root is.
template <typename RootT>
static Type findRefusedTraitType(RootT root,
                                 llvm::function_ref<bool(Type)> admitted) {
  Type refused;
  root.walk([&](Type sub) -> WalkResult {
    if (sub.getDialect().getNamespace() == TraitDialect::getDialectNamespace() &&
        !admitted(sub)) {
      refused = sub;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return refused;
}

/// The erase step's exit check, run with every template still standing.
///
/// Erase deletes nothing for being a template; the collector the pass runs
/// after this check takes what nothing names. What this judges is therefore
/// everything standing *outside* a template, which is final for the stage: it
/// carries no claim, no projection, no generic, and no trait type in a value
/// position, and it names no template. A declaration whose spelling still carries
/// a generic is itself a template, skipped here and collected with the rest.
///
/// Reading the module while the templates stand is stricter than reading what
/// survives collection: a mention of a template from outside one is the defect
/// whoever mentions it, and after collection there would be nothing left to
/// name. Every refusal is a diagnosis at the op that carries it; nothing here
/// answers a defect by deleting what exhibits it.
static LogicalResult checkNothingOutsideATemplateCarriesTheory(ModuleOp module) {
  bool clean = true;

  // The names a symbol use spelled in the module body can resolve to through
  // its root reference, which is what the last clause reads for.
  DenseSet<StringAttr> templateNames;
  for (Operation &op : *module.getBody())
    if (isTemplate(&op))
      templateNames.insert(SymbolTable::getSymbolName(&op));

  auto admitNothing = [](Type) { return false; };

  module.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (isTemplate(op)) {
      // A template is private from birth, so a public one is a template
      // collection may not take. The trait, impl, and proof declarations answer
      // the same law in their own verifiers; every other template is a plain
      // symbol declaration -- a polymorphic function, or a generic definition of
      // a dialect this one does not name -- and is judged here, where the walk
      // meets it, reaching one standing inside a nested symbol table too.
      if (!isa<TraitOp, ImplOp, ProofOp>(op) &&
          SymbolTable::getSymbolVisibility(op) == SymbolTable::Visibility::Public) {
        op->emitOpError()
            << "is a public template: a template is private from birth, so "
               "that nothing outside its own symbol table may name it once "
               "its instances are cut";
        clean = false;
      }
      return WalkResult::skip();
    }

    Type refused;
    auto scan = [&](Type ty) {
      if (!refused)
        refused = findRefusedTraitType(ty, admitNothing);
    };
    for (Type ty : op->getOperandTypes())
      scan(ty);
    for (Type ty : op->getResultTypes())
      scan(ty);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          scan(argument.getType());
    if (!refused)
      refused = findRefusedTraitType(op->getAttrDictionary(), admitNothing);
    if (refused) {
      op->emitOpError() << "still carries " << refused
                        << " after erasure: nothing standing outside a "
                           "template may carry the trait type system";
      clean = false;
    }
    return WalkResult::advance();
  });

  // A symbol use naming a template from outside one keeps that template alive
  // through collection, so it is refused at the op that spells it. Reading the
  // body region walks every use the module's own scope holds without entering a
  // nested symbol table, so an impl's own interior is out of the universe; a
  // use standing in a template that is not a symbol table -- a proof, a
  // polymorphic function -- is reached and dropped by the isForeign filter
  // below.
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(&module.getBodyRegion());
  if (!uses) {
    module.emitOpError() << "carries an operation whose symbol uses cannot be "
                            "read, so no template can be shown unnamed";
    return failure();
  }
  for (const SymbolTable::SymbolUse &use : *uses) {
    if (!templateNames.contains(use.getSymbolRef().getRootReference()))
      continue;
    if (isForeign(use.getUser()))
      continue;
    use.getUser()->emitOpError()
        << "names the template " << use.getSymbolRef()
        << " from outside a template, which no monomorphic program may do";
    clean = false;
  }

  return success(clean);
}

/// Erases all residual polymorphism from the module.
///
/// Three phases run in sequence, and none of them deletes an operation for
/// being a template: the collector the pass runs after them takes what nothing
/// names.
///
/// Phase 1 (applyPartialConversion): Structural op rewrites that erase
///   SSA values.  Claim types map to zero results (1:0 erasure), so ops
///   that carry claims need their operand lists, indices, and signatures
///   rewritten.  Only applyPartialConversion can do this — it manages
///   the value-level bookkeeping (dropping operands, remapping uses).
///   The tuple dialect adjusts tuple.get indices and tuple.make operands;
///   the func dialect rewrites function signatures and call sites.  Every
///   template is legal and recursively legal, so the driver never enters
///   one: an unused template is neither converted nor asked to legalize.
///
/// Phase 2 (the type sweep): Bulk type rewriting.
///   applyPartialConversion only touches operand/result types on ops
///   matched by patterns.  Types inside attributes (e.g. the body
///   TypeAttr on nominal.def) are invisible to it.  This sweep rewrites
///   the remaining types of every op outside a template; nothing writes
///   into a template, whose spelling is resolved when it is cloned for a
///   concrete instance.  The nominal dialect registers its NominalType
///   name mangling here.
///
/// Phase 3 (the exit check): everything standing outside a template is
///   theory-free and names no template.
///
/// Each dialect contributes to phases 1 and 2 via populateErasePolymorphsPatterns.
static LogicalResult erasePolymorphs(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

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
  TypeConverter opConverter = makeErasePolymorphsConverter();

  // The sweep respells types wherever it reaches them, so it carries the seal:
  // an equality's endpoints are a leaf to it, as they are to every replacer the
  // dialect builds.
  AttrTypeReplacer typeSweep = makeEndpointSealedReplacer();

  // Collect from participating dialects
  RewritePatternSet patterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<MonomorphizationInterface>())
      iface->populateErasePolymorphsPatterns(opConverter, patterns, typeSweep);
  }

  // Add trait dialect's own patterns
  patterns.add<EraseProjectOp, EraseWitnessOp>(ctx);
  patterns.add<EraseCoerceOp>(opConverter, ctx);

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, opConverter);
  populateCallOpTypeConversionPattern(patterns, opConverter);
  populateReturnOpTypeConversionPattern(patterns, opConverter);

  ConversionTarget target(*ctx);
  populateErasePolymorphsLegality(target);

  // Apply Phase 1
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return failure();

  // Phase 2: bulk type rewriting.
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

  // Nothing writes into a template: its spelling is resolved when it is cloned
  // for a concrete instance, not by this sweep, so the walk skips a template
  // whole -- its shell and its interior.
  module->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (isTemplate(op))
      return WalkResult::skip();
    typeSweep.replaceElementsIn(op,
                                /*replaceAttrs=*/true,
                                /*replaceLocs=*/false,
                                /*replaceTypes=*/true);
    return WalkResult::advance();
  });

  // Phase 3: the exit check. Everything standing outside a template is
  // theory-free -- no claims, projections, or coerces -- and names no template,
  // so the collector the pass runs next takes every template whole.
  return checkNothingOutsideATemplateCarriesTheory(module);
}

}

TypeConverter makeErasePolymorphsConverter() {
  // ClaimType maps to zero results (the SSA value carrying the erased proof
  // disappears); every other type converts to itself.
  TypeConverter opConverter;
  opConverter.addConversion([](Type ty) { return ty; });
  opConverter.addConversion([](ClaimType ty, SmallVectorImpl<Type> &out) {
    return success();
  });
  return opConverter;
}

void populateErasePolymorphsLegality(ConversionTarget &target,
                                     bool templatesIllegal) {
  // Mark !trait.claim and !trait.proj as illegal
  target.addIllegalOp<AllegeOp, DeriveOp, ProjectOp, WitnessOp, CoerceOp>();
  // A template leaves with monomorphization, so the pass's own target converts
  // neither it nor its interior: the three declarations are legal and recursively
  // legal, and a function is a template exactly while its signature stays
  // polymorphic. A recursively legal op's interior is never enqueued, so an unused
  // template neither converts nor has to legalize; a monomorphic function answers
  // the same law every other op does. The readiness target instead marks a
  // template illegal, so the step is present while one stands rather than absent
  // while a template holds another step's type standing.
  if (templatesIllegal) {
    target.addIllegalOp<TraitOp, ImplOp, ProofOp>();
  } else {
    target.addLegalOp<TraitOp, ImplOp, ProofOp>();
    target.markOpRecursivelyLegal<TraitOp, ImplOp, ProofOp>();
  }
  target.addDynamicallyLegalOp<func::FuncOp>([templatesIllegal](func::FuncOp func) {
    bool isTemplateFunc = isPolymorphicType(Type(func.getFunctionType()));
    bool clean = !opMentionsType<ClaimType>(func) &&
                 !opMentionsType<ProjectionType>(func);
    // a template function is illegal in the readiness target, legal (and
    // recursively legal, below) in the pass's own
    if (templatesIllegal)
      return !isTemplateFunc && clean;
    return isTemplateFunc || clean;
  });
  if (!templatesIllegal)
    target.markOpRecursivelyLegal<func::FuncOp>([](Operation *op) {
      return isPolymorphicType(Type(cast<func::FuncOp>(op).getFunctionType()));
    });
  target.markUnknownOpDynamicallyLegal(
      [templatesIllegal](Operation *op) -> std::optional<bool> {
        // An operation still pending instantiation is not erase's yet: it has no
        // opinion on it, so the readiness walk holds erase behind instantiate and
        // the partial conversion leaves it for a later pass. By the time erase
        // runs, instantiation has resolved every pending op, so this never leaves
        // one standing.
        if (isPendingOp(op))
          return std::nullopt;
        // A template -- including a generic symbol declaration another dialect
        // owns, whose body may still name a claim or projection -- is carried to
        // no target by erasure and cut by the collector. The pass's target leaves
        // it legal whatever theory its spelling still names; the readiness target
        // marks it illegal so the step is present while it stands. Every other op
        // is legal once it carries no claim or projection.
        if (isTemplate(op))
          return templatesIllegal ? std::optional<bool>(false)
                                  : std::optional<bool>(true);
        return !opMentionsType<ClaimType>(op) && !opMentionsType<ProjectionType>(op);
      });
}

bool isRewritableGenericCall(Operation *op) {
  // The single readiness law the two call-lowering patterns gate on, so the
  // patterns and the instantiate step's discharge read one spelling. A
  // trait.func.call is rewritable when its operands are monomorphic, its operand
  // claims proven, it stands at module scope (func.call requires the callee in
  // the same symbol table), and its callee has a signature to specialize
  // against. A trait.method.call is rewritable when its operands are
  // monomorphic, its receiver claim proven, its argument claims proven, and its
  // method has a signature; it lowers in place to a trait.func.call and needs no
  // scope of its own. None of these reads raises a demand. The
  // substitution the patterns then run may still find a callee whose result
  // stays polymorphic -- a call whose instantiation the arguments do not
  // determine -- and refuse it fail-closed; that is the rewrite discovering
  // non-instantiability, not a readiness the program well-formed enough to
  // reach here can fail.
  auto operandsMonomorphic = [](ValueRange operands) {
    for (Value operand : operands)
      if (isPolymorphicType(operand.getType()))
        return false;
    return true;
  };
  auto operandClaimsProven = [](ValueRange operands) {
    for (Value operand : operands)
      if (auto claim = dyn_cast<ClaimType>(operand.getType()))
        if (claim.isApplication() && claim.isMonomorphic() && !claim.isProven())
          return false;
    return true;
  };
  auto atModuleScope = [](Operation *op) {
    Operation *nearestTable = SymbolTable::getNearestSymbolTable(op);
    return nearestTable && isa<ModuleOp>(nearestTable);
  };
  if (auto call = dyn_cast<FuncCallOp>(op))
    return operandsMonomorphic(call.getOperands()) &&
           operandClaimsProven(call.getOperands()) && atModuleScope(op) &&
           succeeded(call.getCalleeFunctionType());
  if (auto call = dyn_cast<MethodCallOp>(op))
    return operandsMonomorphic(call.getOperands()) &&
           call.getClaimType().isProven() &&
           operandClaimsProven(call.getArguments()) &&
           succeeded(call.getMethodFunctionType());
  return false;
}

// One op's share of the two leftover checks' scan: an op one of whose result or
// block-argument types carries a standing obligation -- an unproven monomorphic
// application claim or an unresolved ground projection the leftover checks refuse --
// and that stands outside a template. Reused so the pending-op predicate reads the
// same demand they do. The type shape is tested before the template-ancestor walk,
// so an op carrying no such type never pays for the walk. File-local: only isPendingOp
// reads it.
static bool opCarriesStandingObligation(Operation *op) {
  auto carriesObligation = [&] {
    for (Type t : op->getResultTypes())
      if (typeCarriesStandingObligation(t))
        return true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (typeCarriesStandingObligation(arg.getType()))
            return true;
    return false;
  };
  return carriesObligation() && !isForeign(op);
}

bool isPendingOp(Operation *op) {
  // One op's share of isPendingExpansion, spelled once: an op outside a template is
  // pending instantiation when it is a generic call a pattern can rewrite or it
  // carries a standing obligation -- an unproven monomorphic application claim or an
  // unresolved ground projection the two leftover checks refuse. The op and type shape
  // is tested before the template-ancestor walk: an op that could never be pending pays
  // for no walk, and one that could pays for it once. The instantiate step's qualified
  // discharge reads this to count an operation exactly where a lowering pattern would
  // fire on it, and the erase step's gate reads it to hold while any such op stands, so
  // the two steps share one definition of pending work.
  if (isRewritableGenericCall(op))
    return !isForeign(op);
  return opCarriesStandingObligation(op);
}

bool isPendingExpansion(ModuleOp module) {
  // Instantiation is pending while some op outside a template is pending: a
  // generic call the patterns can rewrite, or an op carrying a standing obligation
  // -- an unproven monomorphic application claim or an unresolved ground
  // projection it has not yet discharged (the demands the two leftover checks
  // refuse). A foreign op is a template or code inside one, which instantiation
  // carries to no target.
  bool pending = false;
  module.walk([&](Operation *op) {
    if (isPendingOp(op)) {
      pending = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return pending;
}

void ErasePolymorphsPass::runOnOperation() {
  if (failed(erasePolymorphs(getOperation())))
    return signalPassFailure();

  // Retiring the templates is one transformation whose interior is not a
  // program: while a template stands beside the definitions the type sweep
  // respelled, it names a generic definition another dialect's erasure took and
  // keeps a spelling the sweep gave the module, so the module between the two
  // halves does not verify. Collection is therefore run here rather than from a
  // slot of its own, and this pass's exit -- with every template nothing names
  // taken -- is the verifiable boundary.
  OpPassManager collect(ModuleOp::getOperationName());
  collect.addPass(createSymbolDCEPass());
  if (failed(runPipeline(collect, getOperation())))
    return signalPassFailure();
}

std::unique_ptr<Pass> createErasePolymorphsPass() {
  return std::make_unique<ErasePolymorphsPass>();
}


} // end mlir::trait
