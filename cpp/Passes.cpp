#include "Instantiation.hpp"
#include "ImplResolution.hpp"
#include "Passes.hpp"
#include "TraitOps.hpp"
#include "Trait.hpp"
#include "TraitTypes.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::trait {

//===----------------------------------------------------------------------===//
// EmitPolymorphsPass
//===----------------------------------------------------------------------===//

LogicalResult emitPolymorphs(ModuleOp module) {
  MLIRContext* ctx = module.getContext();

  RewritePatternSet patterns(ctx);

  // collect patterns from participating dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<ConvertToTraitPatternInterface>())
      iface->populateEmitPolymorphsPatterns(patterns);
  }

  // apply patterns
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  return success();
}

void EmitPolymorphsPass::runOnOperation() {
  if (failed(emitPolymorphs(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createEmitPolymorphsPass() {
  return std::make_unique<EmitPolymorphsPass>();
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

    for (TraitOp v : u.getRequiredTraits()) {
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

struct RewriteTypesInPlace : RewritePattern {
  AttrTypeReplacer &replacer;

  RewriteTypesInPlace(MLIRContext* ctx, AttrTypeReplacer& replacer)
    : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx), replacer(replacer) {}

  bool needsRewrite(Operation *op) const {
    // op needs to be rewritten if replacer would modify any element in op

    // check results
    for (Type t : op->getResultTypes()) {
      if (t != replacer.replace(t)) return true;
    }
    // check attributes
    for (NamedAttribute na : op->getAttrs()) {
      if (na.getValue() != replacer.replace(na.getValue())) return true;
    }
    // region block args
    for (Region &r : op->getRegions()) {
      for (Block &b : r) {
        for (BlockArgument a : b.getArguments()) {
          if (a.getType() != replacer.replace(a.getType())) return true;
        }
      }
    }
    return false;
  }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (!needsRewrite(op)) return rewriter.notifyMatchFailure(op, "no type changes");

    rewriter.modifyOpInPlace(op, [&] {
      // replace attributes, results, & nested block-arg types
      replacer.replaceElementsIn(op,
                                 /*replaceAttrs=*/true,
                                 /*replaceLocs=*/false,
                                 /*replaceTypes=*/true);
    });

    return success();
  }
};

static LogicalResult applySubstitutionInPlace(const DenseMap<Type,Type>& subst, Operation* root) {
  if (subst.empty()) return success();

  AttrTypeReplacer replacer = makeTypeReplacerFromSubstitution(subst);

  MLIRContext *ctx = root->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<RewriteTypesInPlace>(ctx, replacer);

  if (failed(applyPatternsGreedily(root, std::move(patterns))))
    return failure();

  return mlir::verify(root);
}

struct ProveClaimPattern : OpRewritePattern<AllegeOp> {
  using OpRewritePattern::OpRewritePattern;

  // one ImplResolver per module; owned by the pass, passed by ref into this pattern
  ImplResolver& resolver;

  ProveClaimPattern(MLIRContext* ctx,
                    ImplResolver &resolver)
    : OpRewritePattern<AllegeOp>(ctx), resolver(resolver) {}

  LogicalResult matchAndRewrite(AllegeOp op, PatternRewriter& rewriter) const override {
    auto errFn = [&] { return op.emitOpError(); };

    // build or reuse canonical evidence for this claim
    auto sym = resolver.resolveAndEnsureProofFor(op.getClaim(), rewriter, errFn);
    if (failed(sym))
      return rewriter.notifyMatchFailure(op, "couldn't find proof of this claim");

    // replace the allegation with a witness
    rewriter.replaceOpWithNewOp<WitnessOp>(
      op,
      *sym,
      op.getClaim().getTraitApplication()
    );

    return success();
  }
};

} // end namespace

FailureOr<ImplResolver> resolveImpls(ModuleOp module) {
  // emit polymorphs
  if (failed(emitPolymorphs(module)))
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
    patterns.add<ProveClaimPattern>(ctx, resolver);

    // rewrite trait.allege -> trait.witness
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return failure();
  }

  // assert that no trait.allege remain
  bool hasLeftovers = false;
  module.walk([&](AllegeOp op) {
    hasLeftovers = true;
    op.emitError() << "unresolved trait.allege after ProveClaimPattern";
  });
  if (hasLeftovers) return failure();

  // rewrite all proven !trait.claim types to ensure they carry proofs
  if (failed(applySubstitutionInPlace(resolver.buildClaimSubstitutionFromMemo(), module)))
    return failure();

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

namespace {

struct FuncCallOpLowering : public OpRewritePattern<FuncCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncCallOp callOp, PatternRewriter &rewriter) const override {
    // if any of the call's operand types are polymorphic, this call can't be resolved yet
    for (auto op : callOp.getOperands()) {
      if (isPolymorphicType(op.getType()))
        return failure();
    }

    // instantiate the callee
    auto callee = callOp.getOrInstantiateCallee(rewriter);

    // replace with a func.call to the instanced callee
    rewriter.replaceOpWithNewOp<func::CallOp>(
      callOp,
      callee.getSymName(),
      callOp.getResultTypes(),
      callOp.getOperands()
    );

    return success();
  }
};

struct MethodCallOpLowering : public OpRewritePattern<MethodCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MethodCallOp op, PatternRewriter &rewriter) const override {
    // all operands must be monomorphic and the claim must be proven
    for (auto o : op.getOperands()) {
      if (isPolymorphicType(o.getType()))
        return rewriter.notifyMatchFailure(op, "operands are still polymorphic");
    }
    if (!op.getClaimType().isProven()) {
      return rewriter.notifyMatchFailure(op, "claim is still unproven");
    }

    // build the call's substitution
    auto subst = op.buildSubstitution();
    if (failed(subst))
      return rewriter.notifyMatchFailure(op, "couldn't build substitution for call");
    
    // apply subst to result types; all results must be monomorphic
    SmallVector<Type> concreteResults;
    for (Type r : op.getResultTypes()) {
      Type newR = applySubstitutionToFixedPoint(*subst, r);
      if (isPolymorphicType(newR))
        return rewriter.notifyMatchFailure(op, "result type is still polymorphic");
      concreteResults.push_back(newR);
    }

    // get the callee
    func::FuncOp callee = op.getOrInstantiateCallee(rewriter);
    if (!callee) 
      return rewriter.notifyMatchFailure(op, "couldn't get or instantiate callee");

    // pass the claim as the first argument to the instantiated callee
    SmallVector<Value> args;
    args.push_back(op.getClaim());
    llvm::append_range(args, op.getArguments());

    // replace with a trait.func.call to the instantiated callee
    rewriter.replaceOpWithNewOp<FuncCallOp>(
      op,
      concreteResults,
      callee.getSymName(),
      args
    );

    return success();
  }
};

}

LogicalResult instantiateMonomorphs(ModuleOp module) {
  // resolve impls first
  auto resolver = resolveImpls(module);
  if (failed(resolver))
    return failure();

  MLIRContext* ctx = module.getContext();

  // rewrite trait.func.call & trait.method.call
  RewritePatternSet patterns(ctx);
  patterns.add<FuncCallOpLowering,MethodCallOpLowering>(ctx);

  // collect convert-to-trait patterns from other dialects
  for (Dialect *d : ctx->getLoadedDialects()) {
    if (auto *iface = d->getRegisteredInterface<ConvertToTraitPatternInterface>())
      iface->populateInstantiateMonomorphsPatterns(patterns);
  }

  // apply patterns
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  return success();
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

static LogicalResult eraseClaims(ModuleOp module) {
  MLIRContext* ctx = module.getContext();
  ConversionTarget target(*ctx);

  // all trait.project and trait.witness ops are illegal
  target.addIllegalOp<ProjectOp, WitnessOp>();

  // otherwise, an op is legal if it does not mention a !trait.claim type
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return !opMentionsType<ClaimType>(op);
  });

  // create a TypeConverter to erase !trait.claim types
  TypeConverter tc;
  tc.addConversion([](Type ty) { return ty; });
  tc.addConversion([](ClaimType ty, SmallVectorImpl<Type> &out) {
    // leaving out unchanged means erase this type
    return success();
  });

  // erase all trait.project & trait.witness ops
  RewritePatternSet patterns(ctx);
  patterns.add<EraseProjectOp, EraseWitnessOp>(ctx);

  // collect erase claims patterns from other dialects
  for (Dialect *dialect : ctx->getLoadedDialects()) {
    if (auto *iface = dialect->getRegisteredInterface<ConvertToTraitPatternInterface>()) {
      iface->populateEraseClaimsPatterns(tc, patterns);
    }
  }

  // populate conversion patterns for func dialect ops
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, tc);
  populateCallOpTypeConversionPattern(patterns, tc);
  populateReturnOpTypeConversionPattern(patterns, tc);

  return applyPartialConversion(module, target, std::move(patterns));
}

}

LogicalResult monomorphize(ModuleOp module) {
  // instantiate monomorphs first
  if (failed(instantiateMonomorphs(module)))
    return failure();

  // erase polymorphic functions
  for (func::FuncOp f : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (isPolymorphicType(f.getFunctionType()))
      f.erase();
  }

  // erase trait.proof ops
  for (ProofOp proof : llvm::make_early_inc_range(module.getOps<ProofOp>())) {
    proof.erase();
  }

  // erase trait.impl ops
  for (ImplOp impl : llvm::make_early_inc_range(module.getOps<ImplOp>())) {
    impl.erase();
  }

  // erase trait.trait ops
  for (TraitOp trait : llvm::make_early_inc_range(module.getOps<TraitOp>())) {
    trait.erase();
  }

  // erase claims
  // we do this last because all of the above may
  // use !trait.claim, trait.witness, or trait.project
  if (failed(eraseClaims(module)))
    return failure();

  return module.verify();
}

void MonomorphizePass::runOnOperation() {
  if (failed(monomorphize(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createMonomorphizePass() {
  return std::make_unique<MonomorphizePass>();
}

} // end mlir::trait
