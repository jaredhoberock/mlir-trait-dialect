#include "Dialect.hpp"
#include "Instantiation.hpp"
#include "Ops.hpp"
#include "Passes.hpp"
#include "Types.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::trait {

LogicalResult verifyAcyclicTraits(ModuleOp module) {
  enum class Status : uint8_t { NotSeen = 0, InPath, Done };
  DenseMap<TraitOp, Status> status;
  SmallVector<TraitOp, 16> stack;

  std::function<LogicalResult(TraitOp)> dfs = [&](TraitOp u) -> LogicalResult {
    Status &s = status[u];
    if (s == Status::InPath) {
      // back-edge: report the cycle u ... u
      auto it = llvm::find(stack, u);
      auto diag = u.emitError("cycle in trait `given` clause: ");
      for (auto i = it; i != stack.end(); ++i)
        diag << "@" << i->getSymName() << " -> ";
      diag << "@" << u.getSymName();
      return failure();
    }

    if (s == Status::Done) return success();

    s = Status::InPath;
    stack.push_back(u);

    for (TraitOp v : u.getPrereqTraits()) {
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

namespace {

struct ProveClaimPattern : OpRewritePattern<ClaimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ClaimOp claim, PatternRewriter& rewriter) const override {
    // emit claims for each prerequisite
    SmallVector<Value> prereqClaims;
    for (auto prereqApp : claim.getPrereqTraitApplications()) {
      auto newClaim = rewriter.create<ClaimOp>(claim.getLoc(), prereqApp);
      prereqClaims.push_back(newClaim.getResult());
    }

    // replace the claim with a witness
    rewriter.replaceOpWithNewOp<WitnessOp>(
      claim,
      claim.getResult().getType(),
      prereqClaims
    );

    return success();
  }
};

LogicalResult proveClaims(ModuleOp module) {
  // verify traits are acyclic first
  if (failed(verifyAcyclicTraits(module)))
    return failure();

  // apply rewrite patterns
  RewritePatternSet patterns(module.getContext());
  patterns.add<ProveClaimPattern>(module.getContext());
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  return module.verify();
}

}

void ProveClaimsPass::runOnOperation() {
  if (failed(proveClaims(getOperation())))
    signalPassFailure();
}

std::unique_ptr<Pass> createProveClaimsPass() {
  return std::make_unique<ProveClaimsPass>();
}

namespace {

struct FuncCallOpLowering : public OpRewritePattern<FuncCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncCallOp callOp, PatternRewriter &rewriter) const override {
    // if any of the call's operand types are symbolic, this call can't be resolved yet
    for (auto op : callOp.getOperands()) {
      if (containsSymbolicType(op.getType()))
        return failure();
    }

    // instantiate the callee
    auto callee = callOp.getOrInstantiateCallee(rewriter);

    // XXX TODO getOrInstantiateCallee is mangling the name of the callee, even though it's monomorphic
    //          because !trait.claim is symbolic now

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

  LogicalResult matchAndRewrite(MethodCallOp methodCallOp, PatternRewriter &rewriter) const override {
    // if any operand is still symbolic, this call can't be resolved yet
    for (auto op : methodCallOp.getOperands()) {
      if (containsSymbolicType(op.getType()))
        return failure();
    }

    func::FuncOp callee = methodCallOp.getOrInstantiateCallee(rewriter);
    if (!callee)
      return methodCallOp.emitOpError() << "couldn't get or instantiate callee '" << methodCallOp.getMethodRef() << "'";

    // pass the claim as the first argument to the instantiated callee
    SmallVector<Value> args;
    args.push_back(methodCallOp.getClaim());
    llvm::append_range(args, methodCallOp.getArguments());

    // replace with a trait.func.call to the instantiated callee
    rewriter.replaceOpWithNewOp<FuncCallOp>(
      methodCallOp,
      methodCallOp.getResultTypes(),
      callee.getSymName(),
      callee.getFunctionType(),
      args
    );

    return success();
  }
};

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
  
  // populate conversion patterns for func dialect ops
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, tc);
  populateCallOpTypeConversionPattern(patterns, tc);
  populateReturnOpTypeConversionPattern(patterns, tc);
  
  return applyPartialConversion(module, target, std::move(patterns));
}

}

LogicalResult monomorphize(ModuleOp module) {
  // prove claims first
  if (failed(proveClaims(module)))
    return failure();

  MLIRContext* ctx = module.getContext();

  // rewrite trait.func.call & trait.method.call
  {
    RewritePatternSet patterns(ctx);
    patterns.add<FuncCallOpLowering,MethodCallOpLowering>(ctx);

    // collect patterns from other dialects
    for (Dialect *dialect : ctx->getLoadedDialects()) {
      if (auto *iface = dialect->getRegisteredInterface<ConvertToTraitPatternInterface>()) {
        iface->populateConvertToTraitConversionPatterns(patterns);
      }
    }

    // apply patterns
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return failure();
  }

  // erase polymorphic functions
  for (func::FuncOp f : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (isPolymorph(f))
      f.erase();
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
