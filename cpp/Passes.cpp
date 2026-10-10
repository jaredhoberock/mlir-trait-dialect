// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include <llvm/ADT/ScopeExit.h>
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
    // A symbol lookup scope holds what a symbol table answered, so taking a
    // symbol out of the IR is what takes those answers back.
    if (op->hasAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      forgetHeldSymbols();
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
/// a template whole -- its shell and its interior.
static void collectRewritableOpsIn(Operation *root,
                                   SmallVectorImpl<Operation *> &ops) {
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op))
      return WalkResult::skip();
    ops.push_back(op);
    return WalkResult::advance();
  });
}

/// The ops in `module` a rewrite driver may reach, in the module's own order.
static SmallVector<Operation *> collectRewritableOps(ModuleOp module) {
  SmallVector<Operation *> ops;
  for (Operation &op : *module.getBody())
    collectRewritableOpsIn(&op, ops);
  return ops;
}

/// Runs `patterns` greedily over the ops `module` reaches (collectRewritableOps),
/// then over the module-level ops each iteration wrote to, until an iteration
/// writes to none. `ExistingOps` strictness admits nothing a run creates, so a
/// clone is invisible to the iteration that made it and the iteration after is
/// what reaches it; the default strictness would follow a rewritten producer
/// into a region, and `ExistingAndNewOps` would enqueue every op a clone's
/// construction creates. A function no iteration wrote to stands where a
/// previous iteration drove it to a fixed point. What a pattern reads besides
/// the op itself is impl selection, whose every answer is a function of the
/// concrete types asked about and is asked for by the op that needs it, so no
/// fact minted elsewhere changes what a pattern would do to an op no iteration
/// wrote to; carrying only the written functions forward is the fixed point
/// reached proportionally to the work rather than to the module. The rewrite
/// budget spans the whole run: the listener counts applications across
/// iterations, each iteration receives the remainder, and an exhausted
/// remainder or a non-converged iteration fails as one whole-module run does.
static LogicalResult applyPatternsOverReachableOps(ModuleOp module,
                                                   RewritePatternSet &&patterns,
                                                   GreedyRewriteConfig config) {
  FrozenRewritePatternSet frozen(std::move(patterns));
  RewriteEventCounts events;
  events.recordWritesUnder(module.getBody());
  config.setListener(&events);
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);
  config.setScope(&module.getBodyRegion());
  // The stage retypes a claim in place once its proof is established, so
  // between two rewrites a value and a user whose type must equal it -- a
  // select's arms, a block argument, a return -- can name different proofs.
  // The folder requires a replacement of the result's own type, which that
  // interval does not offer; folding is left to the steps after the stage,
  // which read the module it converged to.
  config.enableFolding(false);
  int64_t budget = config.getMaxNumRewrites();

  LogicalResult result = success();
  bool firstIteration = true;
  while (succeeded(result)) {
    SmallVector<Operation *> ops;
    if (firstIteration) {
      ops = collectRewritableOps(module);
    } else {
      // The module-level ops the previous iteration wrote to are the whole of
      // the work left for this one: a clone it minted, and every function a
      // rewrite landed in, whose neighbouring ops read what that rewrite
      // produced.
      for (Operation *op : events.takeWrittenModuleLevelOps(module.getBody()))
        collectRewritableOpsIn(op, ops);
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
    if (!iterationChanged)
      break;
  }
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

/// Verify that every proven claim spelled in a top-level function signature
/// names evidence whose declaration carries to it. A `by @proof` in a declared
/// type is otherwise checked nowhere until a call reaches it, so a signature can
/// name a proof that does not specialize to its claim and go undiagnosed. What
/// that proof cites underneath was decided at the proof op holding it. Only
/// module-level `func.func` signatures are walked; signatures nested inside
/// trait/impl method bodies are not yet covered.
LogicalResult verifyDeclaredClaimProofs(ModuleOp module) {
  LogicalResult status = success();
  for (auto f : module.getOps<func::FuncOp>()) {
    auto errFn = [&] {
      return f.emitOpError() << "declared claim in signature has an invalid proof: ";
    };
    auto byGroundLookup = [&](Type ty) -> FailureOr<Type> {
      return resolveProjectionsByLookup(ty, module, LookupScope::Ground,
                                        /*emitError=*/nullptr);
    };
    if (failed(verifyCitationsIn(Type(f.getFunctionType()), module,
                                 byGroundLookup, errFn)))
      status = failure();
  }
  return status;
}

/// Refuses a proof whose derivation stands too deep. A derivation is followed
/// from each proof through the proof each application premise names, at the
/// instance the proof cites it, and its depth is counted and refused as impl
/// selection counts and refuses an obligation chain (Rust's trait solver's
/// overflow): a proof citing itself, or a ring of proofs citing one another,
/// at ever larger arguments reaches a new application at every step. A proof
/// cited again at an application on the chain is a coinductive citation and
/// counts zero.
///
/// The height below a (proof, application) pair is kept and read where the
/// pair is reached again, and followed anew where the chain reaching it makes
/// it too deep, so a refusal names a real chain. A height that counts pairs
/// above it on the chain as zero holds while those pairs stand there, so it is
/// kept with them and read only while each still stands at its depth (rustc's
/// provisional cache); one that counts none holds on every chain. A height
/// read is then never below the one a fresh walk computes, and the verdict
/// depends on no order of proofs or premises.
static LogicalResult verifyProofDerivationsEnd(ModuleOp module) {
  using Followed = std::pair<Operation *, TraitApplicationAttr>;
  // A height, with the pairs above it on the chain, and their depths there,
  // that the derivation below cites coinductively.
  struct Height {
    unsigned value;
    SmallVector<std::pair<Followed, unsigned>, 1> cites;
  };
  SmallVector<ObligationFrame> chain;
  DenseMap<Followed, unsigned> depthOnChain;
  DenseMap<Followed, Height> heights;
  auto standsWhereCited = [&](const Height &height) {
    return llvm::all_of(height.cites, [&](const auto &cited) {
      auto onChain = depthOnChain.find(cited.first);
      return onChain != depthOnChain.end() && onChain->second == cited.second;
    });
  };
  std::function<FailureOr<Height>(ProofOp, ClaimType)> follow =
      [&](ProofOp proof, ClaimType at) -> FailureOr<Height> {
    TraitApplicationAttr app = at.getTraitApplication();
    Followed key{proof, app};
    if (auto onChain = depthOnChain.find(key); onChain != depthOnChain.end())
      return Height{0, {{key, onChain->second}}};
    // The deepest frame below the pair stands `height - 1` below it.
    if (auto known = heights.find(key);
        known != heights.end() && standsWhereCited(known->second) &&
        chain.size() + known->second.value - 1 < kInstantiationDepthLimit)
      return known->second;
    if (failed(checkObligationChainDepth(chain))) {
      emitObligationOverflow(proof.getLoc(), app, chain);
      return failure();
    }
    // A citation the proof's verifier refuses is refused there.
    auto premises = proof.getPremisesAt(at, /*err=*/nullptr);
    if (failed(premises))
      return Height{1, {}};
    unsigned depth = chain.size();
    chain.push_back(
        {app, FlatSymbolRefAttr::get(proof.getContext(), proof.getSymName())});
    depthOnChain[key] = depth;
    auto popped = llvm::scope_exit([&] {
      chain.pop_back();
      depthOnChain.erase(key);
    });
    Height below{0, {}};
    for (ClaimType premise : *premises) {
      if (!premise.isApplication() || !premise.isProven())
        continue;
      if (auto cited = lookupSymbolFrom<ProofOp>(module, premise.getProof())) {
        FailureOr<Height> height = follow(cited, premise);
        if (failed(height))
          return failure();
        below.value = std::max(below.value, height->value);
        for (const auto &above : height->cites)
          if (above.second < depth && !llvm::is_contained(below.cites, above))
            below.cites.push_back(above);
      }
    }
    ++below.value;
    heights[key] = below;
    return below;
  };
  for (ProofOp proof : module.getOps<ProofOp>())
    if (failed(follow(proof, proof.getProvenClaim())))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// VerifyAcyclicTraitsPass
//===----------------------------------------------------------------------===//

// The structural half of the acyclicity check, which runs before the module is
// verified: it reads trait symbols by name and refuses a dangling requirement
// reference through a diagnostic rather than reaching the aborting trait
// accessor. The trait-to-trait edges it walks form the requirement dependency
// graph; a back-edge is a cycle.
static LogicalResult verifyAcyclicTraitsStructure(ModuleOp module) {
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

    // The requirements are the trait's result signature, so a missing or
    // malformed list is input the screen faces before the full verifier --
    // refuse it rather than read through it.
    ArrayAttr requirements = u.getRequirementsAttr();
    if (!requirements)
      return u.emitError("trait carries no requirements list");

    for (Attribute entry : requirements) {
      // Only application requirements form trait-to-trait edges; an equality
      // requirement has no trait head and cannot close a requirement cycle.
      auto typeAttr = dyn_cast<TypeAttr>(entry);
      auto claim = typeAttr ? dyn_cast<ClaimType>(typeAttr.getValue()) : ClaimType();
      if (!claim || !claim.isApplication())
        continue;
      TraitApplicationAttr app = claim.getTraitApplication();
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

  return module.verify();
}

void VerifyAcyclicTraitsPass::runOnOperation() {
  if (failed(verifyAcyclicTraits(getOperation())))
    signalPassFailure();
}


//===----------------------------------------------------------------------===//
// InstantiateMonomorphsPass
//===----------------------------------------------------------------------===//

namespace {

/// Proves a claim-producing op and replaces it with a trait.witness, or a
/// projection with the evidence it reads.
///
/// An allegation is proven by selection on its claim, and an alleged equality
/// by resolving its projections through the impls selection chooses for their
/// applications; a derive is proven by the proof whose body it is, never by
/// selecting again; a projection of a proven claim is replaced by the evidence
/// the claim's impl returns at its index (`ProjectOp::inlineEvidence`), which
/// this rule then proves where it stands.
struct ProveClaimResultPattern : public RewritePattern {
  ImplResolver &resolver;
  /// The ops this pattern has named a failure at -- an allegation selection
  /// refuses, an equality whose sides selection resolves to two types -- each
  /// named once: selection answers a refusal alike every time it is asked, and
  /// the driver may reach an op again.
  mutable llvm::DenseSet<Operation *> named;
  /// The claims selection refused that this pattern has named why for, which
  /// the stage's exit walk names nowhere again.
  DenseSet<Type> &namedObligations;
  /// Set once this pattern refuses a claim. The stage that owns it fails on a
  /// refusal: a refused op is left standing, and a stage that succeeded past
  /// it would hand the steps after it a module nothing proved.
  bool &refusedAClaim;

  ProveClaimResultPattern(MLIRContext *ctx, ImplResolver &resolver,
                          DenseSet<Type> &namedObligations,
                          bool &refusedAClaim)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      resolver(resolver), namedObligations(namedObligations),
      refusedAClaim(refusedAClaim) {}

  /// Records that `op`'s claim is refused, answering whether it was not
  /// named before, so each refusal is reported once.
  bool refuse(Operation *op) const {
    refusedAClaim = true;
    return named.insert(op).second;
  }

  /// `errFn` while nothing has been named at `op`, so selection names why it
  /// refuses there the first time it is asked and only then.
  llvm::function_ref<InFlightDiagnostic()>
  firstAsk(Operation *op, llvm::function_ref<InFlightDiagnostic()> errFn) const {
    if (named.contains(op))
      return nullptr;
    return errFn;
  }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter& rewriter) const override {
    if (!isa<AllegeOp, DeriveOp, ProjectOp>(op))
      return failure();

    // In-place retyping by proof propagation can prove a claim out from under
    // its producing op, which leaves that op holding exactly the witness it was
    // going to build: its result type already names both the proof and the
    // application. Build it, rather than leaving behind a producer nothing
    // legalizes wherever a consumer still wants its result.
    auto claim = cast<ClaimType>(op->getResult(0).getType());

    // A projection of a proven source is replaced by the evidence the source's
    // impl returns at its index, which the patterns then prove where it stands:
    // a derive transcribed, an allegation decided, a coercion settled.
    if (auto project = dyn_cast<ProjectOp>(op)) {
      if (!project.getSourceClaim().isProven())
        return rewriter.notifyMatchFailure(op, "waits for its source");
      // Evidence with no base, or read past the depth limit, is never
      // inlined; the exit walk names it.
      if (project.readEvidence().end != EvidenceReading::End::Base)
        return rewriter.notifyMatchFailure(op, "its evidence has no base");
      return project.inlineEvidence(rewriter);
    }

    // An alleged equality is proved here once monomorphic.
    if (claim.isEquality()) {
      auto allegation = dyn_cast<AllegeOp>(op);
      if (!allegation || !claim.isMonomorphic())
        return rewriter.notifyMatchFailure(op, "equality claim proved elsewhere");
      return proveAllegedEquality(allegation, claim.getEqualityAttr(), rewriter);
    }

    // A claim spelled unproven is proven by selection once it is monomorphic;
    // a polymorphic one waits for the instance that grounds it.
    if (!claim.isProven() && !claim.isMonomorphic())
      return rewriter.notifyMatchFailure(op, "polymorphic claim deferred");

    // A derive states its own decision: its claim is proven by the proof whose
    // body is this derive over the evidence its operands carry.
    if (auto derive = dyn_cast<DeriveOp>(op))
      return transcribe(derive, rewriter);

    // The proof the op is spelled with; else the canonical evidence selection
    // builds or reuses for its claim, demanded where the op stands: the proof
    // it names is a symbol its own module resolves, and the impls that may
    // serve it are the ones standing there. A selected proof names the
    // application it was recorded under, which selection spelled with the
    // claim's projections resolved; the producer's source spelling would leave
    // the witness and its proof disagreeing on those projections. A refusal is
    // final, so it is named once.
    Answer<ClaimType> proven = claim;
    if (!claim.isProven()) {
      auto errFn = [&] { return op->emitOpError(); };
      proven = resolver.resolveAndEnsureProofFor(
          claim, SelectionSite::of(op), rewriter, firstAsk(op, errFn));
      if (!proven.isAnswer()) {
        (void)refuse(op);
        namedObligations.insert(Type(claim));
        return rewriter.notifyMatchFailure(op, "couldn't find proof of this claim");
      }
    }
    rewriter.replaceOpWithNewOp<WitnessOp>(op, proven->getProof(),
                                           proven->getTraitApplication());
    return success();
  }

  /// Proves `derive` by the proof whose body it is: the impl it cites, at the
  /// arguments its claim and operands determine, given the proof each
  /// application operand names and the evidence of each equality operand at
  /// those arguments. A proof standing with that body answers; otherwise one is
  /// written. The derive waits while an application operand names no proof or
  /// a projection it spells is not yet resolved, and is refused where an
  /// equality's sides resolve apart.
  LogicalResult transcribe(DeriveOp derive, PatternRewriter &rewriter) const {
    // An operand is settled once an application names its proof and an
    // equality stands on something other than an allegation still to be
    // decided.
    SmallVector<FlatSymbolRefAttr> subproofs;
    for (Value operand : derive.getAssumptions()) {
      auto premise = cast<ClaimType>(operand.getType());
      bool settled = premise.isApplication()
                         ? premise.isProven()
                         : !operand.getDefiningOp<AllegeOp>();
      if (!settled)
        return rewriter.notifyMatchFailure(derive, "waits for its operands");
      if (premise.isApplication())
        subproofs.push_back(premise.getProof());
    }

    auto errFn = [&] { return derive.emitOpError(); };
    ModuleOp scope = getAnchorModule(derive);
    ClaimType claim = derive.getDerivedClaim();
    // The verifier has read the citation already; what it accepted is read
    // again here at the instance's spelling.
    ImplOp impl = derive.getImplOp();
    if (!impl)
      return rewriter.notifyMatchFailure(derive, "cites no impl");
    auto arguments = impl.readCitationArguments(
        claim, derive.getAssumptions().getTypes(), errFn);
    if (failed(arguments)) {
      (void)refuse(derive);
      return rewriter.notifyMatchFailure(derive, "cites its impl at no arguments");
    }
    SmallVector<ClaimType> entries = impl.getWhereClaimsAt(*arguments);

    // The proof names its application with the projections it spells
    // resolved, as selection records one; a projection selection does not
    // resolve leaves the derive standing for the stage's exit walk to name.
    SelectionSite site = SelectionSite::of(derive);
    Answer<Type> resolved =
        resolver.resolveProjectionsIn(Type(claim), site, rewriter);
    if (!resolved.isAnswer())
      return rewriter.notifyMatchFailure(derive, "spells no normal form");
    TraitApplicationAttr app = cast<ClaimType>(*resolved).getTraitApplication();
    if (llvm::any_of(app.getTypeArgs(), carriesUndischargedObligation))
      return rewriter.notifyMatchFailure(derive,
                                         "waits for its arguments' obligations");

    // A proof standing with this body answers, and is read rather than
    // written again.
    if (ClaimType standing = resolver.findProof(scope, impl, app, subproofs)) {
      rewriter.replaceOpWithNewOp<WitnessOp>(derive, standing.getProof(),
                                             standing.getTraitApplication());
      return success();
    }

    // An equality premise is ground here, and a ground equality has one
    // answer: the evidence the steps resolving its projections build.
    SmallVector<SmallVector<ResolutionStep>> equalitySteps;
    for (ClaimType entry : entries) {
      if (!entry.isEquality())
        continue;
      TypeEqualityAttr eq = entry.getEqualityAttr();
      SmallVector<ResolutionStep> steps;
      auto sides = resolver.resolveEquality(eq, site, rewriter, steps,
                                            firstAsk(derive, errFn));
      if (!sides.isAnswer()) {
        named.insert(derive);
        return rewriter.notifyMatchFailure(derive,
                                           "a projection is not resolved");
      }
      if (sides->first != sides->second) {
        if (refuse(derive))
          errFn() << "is given " << entry << ", and impl selection "
                  << "resolves its sides to " << sides->first << " and "
                  << sides->second;
        return rewriter.notifyMatchFailure(
            derive, "selection resolves the sides apart");
      }
      equalitySteps.push_back(std::move(steps));
    }

    Answer<ClaimType> proven =
        resolver.writeProof(scope, impl, app, *arguments, entries, subproofs,
                            equalitySteps, site, rewriter);
    if (!proven.isAnswer()) {
      named.insert(derive);
      return rewriter.notifyMatchFailure(derive,
                                         "a projection is not resolved");
    }
    rewriter.replaceOpWithNewOp<WitnessOp>(derive, proven->getProof(),
                                           proven->getTraitApplication());
    return success();
  }

  /// Proves the monomorphic equality `eq` `allegation` alleges: each projection
  /// its sides spell resolves through the impl selection chooses for the
  /// projection's application, one step at a time, and where both sides reach
  /// one type the allegation becomes the evidence of those steps. Sides that
  /// reach two types are refused where the allegation stands; a projection
  /// selection does not resolve leaves the allegation standing for the stage's
  /// exit walk to name.
  LogicalResult proveAllegedEquality(AllegeOp allegation, TypeEqualityAttr eq,
                                     PatternRewriter &rewriter) const {
    auto errFn = [&] { return allegation.emitOpError(); };
    SmallVector<ResolutionStep> steps;
    auto sides = resolver.resolveEquality(eq, SelectionSite::of(allegation),
                                          rewriter, steps,
                                          firstAsk(allegation, errFn));
    if (!sides.isAnswer()) {
      named.insert(allegation);
      return rewriter.notifyMatchFailure(allegation,
                                         "a projection is not resolved");
    }
    if (sides->first != sides->second) {
      if (refuse(allegation))
        errFn() << "alleges " << eq.getLhs() << " = " << eq.getRhs()
                << ", and impl selection resolves its sides to "
                << sides->first << " and " << sides->second;
      return rewriter.notifyMatchFailure(allegation,
                                         "selection resolves the sides apart");
    }
    rewriter.setInsertionPoint(allegation);
    rewriter.replaceOp(allegation, buildEqualityEvidence(rewriter,
                                                         allegation.getLoc(),
                                                         eq, steps));
    return success();
  }
};

} // end namespace


/// Extend this substitution with bindings that resolve concrete `!trait.proj`
/// types visible after applying the current substitution.
void CallSubstitution::discoverProjectionBindings(
    TypeRange types, ProjectionResolver resolve, bool &declined) {
  for (Type ty : types) {
    apply(ty).walk([&](Type t) {
      auto proj = dyn_cast<ProjectionType>(t);
      if (!proj || isPolymorphicType(proj))
        return;
      if (projectionBindings.lookup(proj))
        return;
      FailureOr<Type> resolved = resolve(proj);
      if (failed(resolved)) {
        declined = true;
        return;
      }
      projectionBindings.bind(proj, *resolved);
    });
  }
}

FailureOr<CallSubstitution> CallSubstitution::forCall(
    SpecializationMap specialization, TypeRange operandTypes,
    TypeRange resultTypes, FunctionType formalTy, ProjectionResolver resolve) {
  CallSubstitution subst(std::move(specialization));

  bool changed;
  bool declined;
  do {
    // The component maps grow monotonically; the raw component sum is not
    // affected by fixed-point normalization of the merged map.
    size_t before = subst.specialization.bindingCount() +
                    subst.projectionBindings.bindingCount();

    // A projection the resolution could not answer in this iteration may be
    // answered by the bindings this iteration goes on to add, so only the last
    // iteration's refusals say what this substitution is missing.
    declined = false;
    subst.discoverProjectionBindings(resultTypes, resolve, declined);
    subst.discoverProjectionBindings(operandTypes, resolve, declined);
    if (formalTy) {
      subst.discoverProjectionBindings(formalTy.getInputs(), resolve, declined);
      subst.discoverProjectionBindings(formalTy.getResults(), resolve,
                                       declined);
    }

    changed = subst.specialization.bindingCount() +
                  subst.projectionBindings.bindingCount() !=
              before;
  } while (changed);

  // A substitution that cannot spell one of the call's projections would
  // specialize the callee against a spelling the projection still stands in,
  // and nothing afterwards revisits a callee already specialized. The call
  // stays standing, with the projection impl selection refused, for the
  // stage's exit walk to name.
  if (declined)
    return failure();
  return subst;
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

/// Impl selection's reading of a type's projections at `site`: each whose head
/// application is ground resolved to its normal form, and a type whose
/// projections have none left spelled, the overflow named where selection met
/// it. Which impl serves a projection is settled by its head, and what that
/// impl binds is a function of the projection's own associated-type arguments,
/// so a call's reading of a variable that stands only inside those arguments
/// reads it off the binding.
static Type readThroughSelection(Type ty, ImplResolver &resolver,
                                 const SelectionSite &site,
                                 PatternRewriter &rewriter) {
  Answer<Type> resolved = resolver.resolveProjectionsIn(
      ty, site, rewriter, makeGroundHeadProjectionReplacer);
  return resolved.isAnswer() ? *resolved : ty;
}

/// The closed call-site substitution of `op` against its callee's signature
/// `formalTy`, read through impl selection in the module the call stands in, on
/// top of the evidence the call itself carries.
template <typename CallOpT>
static FailureOr<CallSubstitution>
buildCallSubstitution(CallOpT op, PatternRewriter &rewriter,
                      ImplResolver &resolver, FunctionType formalTy) {
  SelectionSite site = SelectionSite::of(op);
  auto selection = [&](Type ty) {
    return readThroughSelection(ty, resolver, site, rewriter);
  };
  auto specialization = op.buildParameterSpecialization(selection);
  if (failed(specialization)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't build substitution");
    return failure();
  }
  auto resolve = [&](ProjectionType proj) -> FailureOr<Type> {
    auto resolved = resolver.resolveProjection(proj, site, rewriter);
    if (!resolved.isAnswer())
      return failure();
    return resolved->getBinding();
  };
  return CallSubstitution::forCall(std::move(*specialization),
                                   op.getOperandTypes(), op.getResultTypes(),
                                   formalTy, resolve);
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
                     ImplResolver &resolver, FunctionType formalTy) {
  Operation *caller =
      op.getOperation()->template getParentOfType<func::FuncOp>();
  Attribute templateKey = instantiationTemplateKey(op);
  InstantiationChain &chain = resolver.getInstantiationChain();
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

  auto subst = buildCallSubstitution(op, rewriter, resolver, formalTy);
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

  SelectionSite site = SelectionSite::of(op);
  auto selection = [&](Type ty) {
    return readThroughSelection(ty, resolver, site, rewriter);
  };
  auto callee = op.getOrSpecializeCallee(rewriter, *subst, selection);
  if (failed(callee)) {
    (void)rewriter.notifyMatchFailure(op, "couldn't get or specialize callee");
    return failure();
  }
  target.callee = *callee;

  chain.note(target.callee.getOperation(), caller, templateKey);
  return target;
}

/// Lowers a call of `op`'s callee, which binds no type parameter and so is no
/// template: the call reaches it as written, so what the call supplies must be
/// what it declares, and no specialization is built. A callee cut as an
/// instance can still spell an obligation the cut minted, and an operand's
/// producer or a result can still spell one too; the call waits for each
/// spelling to be settled where it stands, so the two are compared only once
/// each is.
static LogicalResult lowerCallOfNonTemplate(FuncCallOp op,
                                            PatternRewriter &rewriter) {
  func::FuncOp callee = *op.getCallee();
  TypeRange parameters = callee.getFunctionType().getInputs();
  TypeRange operands = op.getOperandTypes();
  if (llvm::any_of(parameters, carriesUndischargedObligation) ||
      llvm::any_of(operands, carriesUndischargedObligation) ||
      llvm::any_of(op.getResultTypes(), carriesUndischargedObligation))
    return rewriter.notifyMatchFailure(op, "an obligation stands unsettled");
  if (parameters.size() != operands.size())
    return op.emitOpError() << "passes " << operands.size()
                            << " operand(s) to '@" << op.getCalleeName()
                            << "', which takes " << parameters.size();
  for (auto [index, types] : llvm::enumerate(llvm::zip(parameters, operands))) {
    auto [parameter, operand] = types;
    if (parameter != operand)
      return op.emitOpError() << "passes " << operand << " as operand #"
                              << index << " to '@" << op.getCalleeName()
                              << "', which takes " << parameter;
  }
  rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getSymName(),
                                            op.getResultTypes(),
                                            op.getOperands());
  return success();
}

/// Lowers a trait call whose instance is ready to a call of that instance.
///
/// A `trait.func.call` becomes a `func.call` of the specialized callee, its
/// operands passing through untouched: the readiness law established that every
/// operand claim is proven, so specialization never bakes an unprovable claim
/// parameter into the callee. A `trait.method.call` becomes a `trait.func.call`
/// of the method's free-function instance, its receiver claim passed as the
/// leading argument.
template <typename CallOpT>
struct CallOpLowering : public OpRewritePattern<CallOpT> {
  ImplResolver &resolver;

  CallOpLowering(MLIRContext *ctx, ImplResolver &resolver)
    : OpRewritePattern<CallOpT>(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(CallOpT op, PatternRewriter &rewriter) const override {
    // The one readiness law, checked before any demand is raised: monomorphic
    // operands, proven claims, and a callee with a signature -- for a free
    // function, one at module scope.
    if (!isRewritableGenericCall(op))
      return rewriter.notifyMatchFailure(op, "not a rewritable generic call");

    // The predicate confirmed the callee's signature exists; read it again to
    // specialize against, as the substitution below reads it.
    FailureOr<FunctionType> formalTy;
    if constexpr (std::is_same_v<CallOpT, FuncCallOp>)
      formalTy = op.getCalleeFunctionType();
    else
      formalTy = op.getMethodFunctionType();
    if (failed(formalTy))
      return rewriter.notifyMatchFailure(op, "couldn't get the callee's function type");

    // A call computing evidence is replaced by the body the receiver's impl
    // wrote for it: the evidence is read by position, never selected again.
    if constexpr (std::is_same_v<CallOpT, MethodCallOp>) {
      if (op.computesEvidence()) {
        auto subst = buildCallSubstitution(op, rewriter, resolver, *formalTy);
        if (failed(subst))
          return failure();
        SelectionSite site = SelectionSite::of(op);
        auto selection = [&](Type ty) {
          return readThroughSelection(ty, resolver, site, rewriter);
        };
        return op.inlineEvidence(rewriter, *subst, selection);
      }
    }

    if constexpr (std::is_same_v<CallOpT, FuncCallOp>) {
      if (op.getCalleeTypeParams().empty())
        return lowerCallOfNonTemplate(op, rewriter);
    }

    auto target = specializeCallTarget(op, rewriter, resolver, *formalTy);
    if (failed(target))
      return failure();

    if constexpr (std::is_same_v<CallOpT, FuncCallOp>) {
      rewriter.replaceOpWithNewOp<func::CallOp>(
          op, target->callee.getSymName(), target->resultTypes, op.getOperands());
    } else {
      SmallVector<Value> args;
      args.push_back(op.getClaim());
      llvm::append_range(args, op.getArguments());
      rewriter.replaceOpWithNewOp<FuncCallOp>(
          op, target->resultTypes, target->callee.getSymName(), args);
    }
    return success();
  }
};

/// Settles a coerce whose types have met: a claim result takes its input's
/// proof, which the coerce verifier holds to its input's -- a coerce changes
/// how a claim is spelled, never the evidence it stands on -- and a coerce
/// whose result is spelled as its input is replaced by the input, the fold the
/// stage's drivers do not run (applyPatternsOverReachableOps). The evidence it
/// cited stays standing for the stage to decide: an allegation or a projection
/// is no dead op while its obligation stands (`ObligationResource`).
struct SettleCoercePattern : public OpRewritePattern<CoerceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CoerceOp coerce,
                                PatternRewriter &rewriter) const override {
    Type input = coerce.getInput().getType();
    if (coerce.getResult().getType() == input) {
      rewriter.replaceOp(coerce, coerce.getInput());
      return success();
    }
    auto result = dyn_cast<ClaimType>(coerce.getResult().getType());
    ClaimType carrying = result && !result.isProven()
                             ? result.carryingProofOf(dyn_cast<ClaimType>(input))
                             : ClaimType();
    if (!carrying)
      return rewriter.notifyMatchFailure(coerce, "its types have not met");
    rewriter.modifyOpInPlace(coerce,
                             [&] { coerce.getResult().setType(carrying); });
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
    // non-ground (mentions a poly var, a projection, or a claim).
    if (llvm::all_of(iface->getResultTypes(), isGroundType))
      return rewriter.notifyMatchFailure(iface, "result types are already ground");

    // only run when all operands are monomorphic
    for (Type ty : iface->getOperandTypes()) {
      if (isPolymorphicType(ty))
        return rewriter.notifyMatchFailure(iface, "operands are still polymorphic");
    }

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

/// Asks impl selection for the obligations an op spells and respells the op
/// with the answers: each unproven monomorphic application claim becomes the
/// claim proven, and each ground projection the type it resolves to.
///
/// An op is the one place its own spelling is settled, so the obligation is
/// asked for by the op that holds it, wherever it stands and whenever the op is
/// reached -- a clone a substitution minted included -- as Rust's
/// monomorphization collector asks `Instance::resolve` per use. Selection
/// memoizes every answer, so every op spelling one claim names one proof. An
/// equality's endpoints state a proposition and are left as spelled. A claim
/// whose evidence its producer reads by position is that reading's to prove,
/// never selection's, so such a result keeps its spelling, as does the result
/// of an op that infers its result types from its operands, whose evidence is
/// theirs (`MonomorphizeResultTypesPattern`); and a call computing evidence
/// keeps its result's projections as well, since the variables of the
/// requirement it computes are read off that spelling when it is replaced by
/// its method's body. What selection refuses stays spelled for the stage's exit
/// walk to name.
struct SettleSpelledObligationsPattern : public RewritePattern {
  ImplResolver &resolver;

  SettleSpelledObligationsPattern(MLIRContext *ctx, ImplResolver &resolver)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
      resolver(resolver) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto call = dyn_cast<MethodCallOp>(op); call && call.computesEvidence())
      return failure();
    bool provesClaims = !producesPositionalEvidence(op) &&
                        !isa<InferTypeOpInterface>(op) &&
                        opMentionsType<ClaimType>(op);
    if (!provesClaims && !opMentionsType<ProjectionType>(op))
      return failure();

    SelectionSite site = SelectionSite::of(op);
    AttrTypeReplacer replacer = makeEndpointSealedReplacer();
    // A ground projection is resolved to its normal form in one rewrite; one
    // that has none is named once, and left as spelled.
    replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
      if (isPolymorphicType(proj))
        return std::nullopt;
      Answer<Type> resolved =
          resolver.resolveProjectionsIn(Type(proj), site, rewriter);
      if (!resolved.isAnswer() || *resolved == Type(proj))
        return std::nullopt;
      return *resolved;
    });
    // The proven spelling names the same application, whose type arguments can
    // spell claims and projections of their own, so the walk continues into the
    // result instead of stopping at it.
    replacer.addReplacement(
        [&](ClaimType claim) -> std::optional<std::pair<Type, WalkResult>> {
          if (!provesClaims || !claim.isApplication() || claim.isProven() ||
              !claim.isMonomorphic())
            return std::nullopt;
          auto proven = resolver.resolveAndEnsureProofFor(claim, site, rewriter);
          if (!proven.isAnswer())
            return std::nullopt;
          return std::make_pair(Type(*proven), WalkResult::advance());
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
    // A call judges its callee's signature, so a respelled callee's calls are
    // asked again.
    if (auto function = dyn_cast<func::FuncOp>(op))
      if (auto uses = SymbolTable::getSymbolUses(function, site.scope))
        for (const SymbolTable::SymbolUse &use : *uses)
          rewriter.modifyOpInPlace(use.getUser(), [] {});
    return success();
  }
};

/// Whether a monomorphic equality claim is settled at the leftover check: its
/// two endpoints ground-resolve to one spelling through impls whose obligations
/// hold. Fails where an endpoint's resolution overflows, the overflow named
/// at `carrier`.
///
/// An equality claim carries no proof -- its evidence is the value
/// itself -- so unlike an application claim it is never "proven"; it is
/// discharged instead when the projections in its endpoints resolve and the two
/// endpoints meet at one ground type. Selection resolves a projection only
/// through an impl whose assumptions are satisfiable and refuses it otherwise,
/// so a projection whose only candidate impl is conditional with an
/// undischarged assumption stays spelled and the endpoints do not meet -- the
/// settlement never resolves through an impl whose where-bounds do not hold.
/// Resolution runs to a fixed point because one hop's binding may spell the
/// next.
static FailureOr<bool>
equalityClaimGroundResolvesToOneSpelling(ClaimType claim,
                                         ImplResolver &resolver,
                                         Operation *carrier,
                                         OpBuilder &builder) {
  auto eq = claim.getEqualityAttr();
  if (!eq)
    return false;
  SelectionSite site = SelectionSite::of(carrier);
  Answer<Type> lhs = resolver.resolveProjectionsIn(eq.getLhs(), site, builder);
  if (!lhs.isAnswer())
    return failure();
  Answer<Type> rhs = resolver.resolveProjectionsIn(eq.getRhs(), site, builder);
  if (!rhs.isAnswer())
    return failure();
  return *lhs == *rhs && isGroundType(*lhs);
}

} // end namespace

LogicalResult instantiateMonomorphs(ModuleOp module) {
  // A symbol name this stage resolves is scanned for once and answered from
  // what was held after that. The stage appends symbols -- the impls it
  // generates, the proofs it records, the instances it cuts -- which leaves
  // what a table already answered standing, and it takes none away except
  // through a rewrite driver, whose listener reports the erasure.
  SymbolLookupScope symbolAnswers;

  // verify traits are acyclic
  if (failed(verifyAcyclicTraits(module)))
    return failure();

  // verify that proofs named in declared signatures actually prove their claims
  if (failed(verifyDeclaredClaimProofs(module)))
    return failure();

  // refuse a proof whose derivation keeps reaching larger applications
  if (failed(verifyProofDerivationsEnd(module)))
    return failure();

  MLIRContext* ctx = module.getContext();
  ImplResolver resolver(module);

  // One driver: prove claim producers (allege, derive, project), settle what
  // each op spells, lower trait.func.call and trait.method.call to instances,
  // and monomorphize any generic op whose results become monomorphic. Each
  // pattern asks impl selection for what it needs where it needs it, and the
  // driver re-walks every function a rewrite landed in, so a fact selection
  // mints reaches every op that reads it.
  //
  // A claim the driver refuses -- a derive given evidence its proof discharges
  // otherwise, an allegation nothing serves -- was named where it stood, and
  // the driver and the walks below run on so that everything else standing is
  // named too.
  bool refusedAClaim = false;
  DenseSet<Type> namedObligations;
  {
    RewritePatternSet patterns(ctx);
    patterns.add<ProveClaimResultPattern>(ctx, resolver, namedObligations,
                                          refusedAClaim);
    patterns.add<SettleSpelledObligationsPattern>(ctx, resolver);
    patterns.add<MonomorphizeResultTypesPattern, SettleCoercePattern>(ctx);
    patterns.add<CallOpLowering<FuncCallOp>, CallOpLowering<MethodCallOp>>(
        ctx, resolver);

    // collect instantiate-monomorphs patterns from other dialects
    for (Dialect *d : ctx->getLoadedDialects()) {
      if (auto *iface = d->getRegisteredInterface<MonomorphizationInterface>())
        iface->populateInstantiateMonomorphsPatterns(patterns);
    }

    GreedyRewriteConfig config;
    config.setMaxNumRewrites(rewriteBudgetFor(module));
    if (failed(applyPatternsOverReachableOps(module, std::move(patterns),
                                             config)))
      return module.emitError(
          "instantiate-monomorphs did not converge: rewrite budget exceeded, "
          "which indicates a non-confluent pattern pair cycling on a type "
          "spelling");
  }

  // A call refused on the instantiation depth limit, an obligation chain past
  // it, and a demand whose projections still change after the limit's worth
  // of steps have each named themselves, and the greedy driver took the
  // refusal for a pattern that did not apply. An overflow is a hard error, as
  // rustc's is: the stage fails here rather than naming what it left standing
  // over a chain it stopped.
  if (resolver.getInstantiationChain().wasLimitReached() ||
      resolver.hasOverflowed())
    return failure();

  // The walks below ask impl selection about what still stands, and selection
  // may generate the impl a resolution chain runs through. A generated impl is
  // a complete template the module verifier checks and nothing below
  // revisits, inserted at the module body where it belongs; the builder's
  // listener is the one selection requires and has nothing to do.
  OpBuilder::Listener settleListener;
  OpBuilder settleBuilder(ctx, &settleListener);

  // An obligation still standing because selection refused it is named with
  // the candidates selection recorded, once, at the first op the walks below
  // reach spelling it: the leftover report says that it stands, not why. A
  // claim the claim-proving pattern was refused was named where it first
  // asked, and a claim a positional reading produces is no question of
  // selection's.
  auto nameRefusal = [&](Operation *op, Type obligation) {
    // A refusal is selection's answer about an application, so a projection
    // is named under its head's claim, and an application named once is not
    // named again.
    auto projection = dyn_cast<ProjectionType>(obligation);
    Type application = projection ? Type(projection.asClaim()) : obligation;
    if (!namedObligations.insert(application).second)
      return;
    if (auto claim = dyn_cast<ClaimType>(obligation))
      if (!claim.isApplication() || producesPositionalEvidence(op))
        return;
    resolver.nameRefusal(obligation, getAnchorModule(op),
                         [&] { return emitError(op->getLoc()); });
  };

  // Assert that no op produced an unproven monomorphic claim that escaped
  // proving, nor takes an unproven application claim as a block argument; an
  // equality a block takes is a hypothesis its predecessors supply. Keying this
  // check on the types values carry rather than on the set of claim-producing
  // ops makes it total over producers: an op whose claims the patterns above
  // fail to discharge is an error here, never a silent gap. The one claim that
  // passes unproven is an equality whose endpoints ground-resolve to one
  // spelling, standing on an op erasure removes. The whole type is walked, so a
  // claim nested inside an aggregate is caught too, not only a claim that is
  // the root type. Trait infrastructure regions are templates and keep their
  // unproven claims. The candidate claims are gathered under the walk and
  // judged after it closes, because judging may insert into the module.
  bool hasLeftovers = false;
  SmallVector<std::pair<Operation *, ClaimType>> monomorphicClaims;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op))
      return WalkResult::skip();
    auto gather = [&](Type root, bool equalities) {
      walkObligationSites(root, [&](Type sub) {
        auto claim = dyn_cast<ClaimType>(sub);
        if (!claim)
          return;
        bool standingEquality = equalities && claim.isEquality() &&
                                claim.isMonomorphic();
        if (standingEquality || isUndischargedObligation(claim))
          monomorphicClaims.emplace_back(op, claim);
      });
    };
    for (Type t : op->getResultTypes())
      gather(t, /*equalities=*/true);
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (Value arg : b.getArguments())
          gather(arg.getType(), /*equalities=*/false);
    return WalkResult::advance();
  });
  for (auto [op, claim] : monomorphicClaims) {
    // An equality claim has no proof to await; it is settled when its endpoints
    // ground-resolve to one spelling through impls whose obligations hold, read
    // in the module the op carrying it stands in. A monomorphic equality that
    // resolves is not a leftover; one whose projection has no
    // obligation-holding impl stays unequal and is reported like an unprovable
    // application claim. An allegation is proved by the claim-proving pattern
    // alone, so one still standing here is a leftover however its endpoints
    // resolve. A `trait.witness` or `trait.project` of an equality that
    // resolves is removed by erasure.
    if (claim.isEquality() && !isa<AllegeOp>(op)) {
      FailureOr<bool> settled = equalityClaimGroundResolvesToOneSpelling(
          claim, resolver, op, settleBuilder);
      if (succeeded(settled) && *settled)
        continue;
      // An endpoint whose resolution overflows is named where it was met, and
      // alone.
      if (failed(settled)) {
        hasLeftovers = true;
        continue;
      }
    }
    hasLeftovers = true;
    // Evidence read past the depth limit is refused as any obligation chain
    // that deep is, and named there alone.
    if (auto project = dyn_cast<ProjectOp>(op);
        project && project.getSourceClaim().isProven()) {
      EvidenceReading reading = project.readEvidence();
      if (reading.end == EvidenceReading::End::Overflow) {
        emitObligationOverflow(op->getLoc(), reading.chain.back().application,
                               reading.chain);
        continue;
      }
    }
    nameRefusal(op, claim);
    InFlightDiagnostic report =
        op->emitError() << "unproven monomorphic claim " << claim
        << " after instantiate-monomorphs";
    // A hop off a proven claim reads its requirement's evidence out of the
    // source's proof, and evidence the proof determines nothing for leaves the
    // hop's claim unproven for selection. Where selection could not prove it
    // either, the evidence the source names is what the report points at.
    if (auto project = dyn_cast<ProjectOp>(op)) {
      if (ClaimType source = project.getSourceClaim(); source.isProven()) {
        EvidenceReading reading = project.readEvidence();
        Diagnostic &note = report.attachNote();
        note << "its source " << source << " names " << source.getProof()
             << ", whose evidence for requirement " << project.getIndex();
        if (reading.end != EvidenceReading::End::Cycle) {
          note << " nothing decides here";
        } else {
          note << " has no base: it is read through the returns of ";
          llvm::interleaveComma(reading.impls, note, [&](StringAttr impl) {
            note << "@" << impl.getValue();
          });
          note << " back to a requirement it stands for";
        }
      }
    }
  }

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
  SmallVector<std::pair<Operation *, ProjectionType>> standingProjections;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op))
      return WalkResult::skip();
    auto gather = [&](Type root) {
      walkObligationSites(root, [&](Type sub) {
        auto proj = dyn_cast<ProjectionType>(sub);
        if (proj && isUndischargedObligation(proj))
          standingProjections.emplace_back(op, proj);
      });
    };
    for (Type t : op->getResultTypes())
      gather(t);
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (Value arg : b.getArguments())
          gather(arg.getType());
    return WalkResult::advance();
  });
  for (auto [op, proj] : standingProjections) {
    nameRefusal(op, proj);
    op->emitError() << "unresolved projection " << proj
                    << " after instantiate-monomorphs";
  }
  if (hasLeftovers || !standingProjections.empty())
    return failure();

  // A ground projection can also stand where the walk above does not read: in
  // an attribute, or in an equality's endpoints, which state a proposition and
  // are left as spelled. One there that selection resolves has a meaning; one
  // selection refuses names a type no impl gives a meaning, which erasure would
  // otherwise take away unread, so it is put to selection here and refused at
  // the op carrying it.
  SmallVector<std::pair<Operation *, ProjectionType>> spelledProjections;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTemplate(op))
      return WalkResult::skip();
    llvm::SetVector<ProjectionType> spelled;
    auto collect = [&](auto root) {
      root.walk([&](ProjectionType proj) {
        if (!isPolymorphicType(Type(proj)))
          spelled.insert(proj);
      });
    };
    for (Type t : op->getResultTypes())
      collect(t);
    for (Region &r : op->getRegions())
      for (Block &b : r)
        for (Value arg : b.getArguments())
          collect(arg.getType());
    collect(Attribute(op->getAttrDictionary()));
    for (ProjectionType proj : spelled)
      spelledProjections.emplace_back(op, proj);
    return WalkResult::advance();
  });
  bool sawUnresolvedProjection = false;
  for (auto [op, proj] : spelledProjections) {
    if (resolver.resolveProjection(proj, SelectionSite::of(op), settleBuilder)
            .isAnswer())
      continue;
    nameRefusal(op, proj);
    op->emitError() << "unresolved projection " << proj
                    << " after instantiate-monomorphs";
    sawUnresolvedProjection = true;
  }
  if (sawUnresolvedProjection)
    return failure();

  // A generic call the driver could still rewrite but did not is a call whose
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

  // ModuleOp's own verifier hook runs here over the module shell -- it does not
  // recurse into the body, and this shallow tail judges no coerce. The bonded
  // erase pass refuses a coerce at its barrier where its endpoints stand apart:
  // they cannot be discharged and cannot cross.
  if (failed(module.verify()))
    return failure();

  // A claim the driver refused was named where it stood. The stage fails on it
  // here: a refusal is an error in the program, and a stage that reported one
  // and then succeeded would let the steps after it run on a module nothing
  // proved.
  return success(!refusedAClaim);
}

void InstantiateMonomorphsPass::runOnOperation() {
  if (failed(instantiateMonomorphs(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createInstantiateMonomorphsPass() {
  return std::make_unique<InstantiateMonomorphsPass>();
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
    // reaches the barrier live is judged here (one dropped as dead earlier,
    // during instantiate-monomorphs, forwarded no value and so needs no
    // judgment). Comparison strips application-claim proofs, exactly as the
    // verifier does: a coerce compares modulo the proof permanently, so
    // exchanging a proof label alone is not a surviving difference. (The
    // value-carrying arm below needs no strip; the values it forwards carry no
    // proof.)
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
  // Every application claim an operand spells names its proof, the claims
  // standing in another claim's arguments included: what an instance is keyed
  // by is the evidence at each position, all of it (`InstanceKey`).
  auto operandClaimsProven = [](ValueRange operands) {
    for (Value operand : operands) {
      bool unproven = false;
      walkObligationSites(operand.getType(), [&](Type sub) {
        unproven = unproven ||
                   (isa<ClaimType>(sub) && isUndischargedObligation(sub));
      });
      if (unproven)
        return false;
    }
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
           operandClaimsProven(call.getOperands()) &&
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
      if (carriesUndischargedObligation(t))
        return true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (carriesUndischargedObligation(arg.getType()))
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
