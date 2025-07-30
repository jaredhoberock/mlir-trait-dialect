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

    // pass the proof as the first argument to the instantiated callee
    SmallVector<Value> args;
    args.push_back(methodCallOp.getProof());
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

static LogicalResult eraseProofs(ModuleOp module) {
  MLIRContext* ctx = module.getContext();
  ConversionTarget target(*ctx);

  // all trait.project and trait.witness ops are illegal
  target.addIllegalOp<ProjectOp, WitnessOp>();

  // otherwise, an op is legal if it does not mention !trait.proof
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return !opMentionsProofType(op);
  });
  
  // create a TypeConverter to erase !trait.proof types
  TypeConverter tc;
  tc.addConversion([](Type ty) { return ty; });
  tc.addConversion([](ProofType ty, SmallVectorImpl<Type> &out) {
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

void MonomorphizePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // apply rewrite patterns
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
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
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

  // erase proofs
  // we do this last because all of the above may
  // use !trait.proof, trait.witness, & trait.project
  if (failed(eraseProofs(module)))
    signalPassFailure();
}

std::unique_ptr<Pass> createMonomorphizePass() {
  return std::make_unique<MonomorphizePass>();
}

} // end mlir::trait
