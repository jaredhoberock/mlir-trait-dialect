#include "Dialect.hpp"
#include "Instantiation.hpp"
#include "Ops.hpp"
#include "Passes.hpp"
#include "Types.hpp"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::trait {

namespace {

struct FuncCallOpLowering : public OpRewritePattern<FuncCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncCallOp callOp, PatternRewriter &rewriter) const override {
    // if any of the call's operand types are symbolic, this call can't be resolved yet
    for (auto op : callOp.getOperands()) {
      if (isa<SymbolicTypeInterface>(op.getType()))
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
    // if the receiver type or any of the call's operand types are symbolic, this call can't be resolved yet
    if (isa<SymbolicTypeInterface>(methodCallOp.getReceiverType()))
      return failure();

    for (auto op : methodCallOp.getOperands()) {
      if (isa<SymbolicTypeInterface>(op.getType()))
        return failure();
    }

    func::FuncOp callee = methodCallOp.getOrInstantiateCallee(rewriter);
    if (!callee)
      return methodCallOp.emitOpError("couldn't get or instantiate callee");

    // replace with a trait.func.call to the instantiated callee
    rewriter.replaceOpWithNewOp<FuncCallOp>(
      methodCallOp,
      methodCallOp.getResultTypes(),
      callee.getSymName(),
      callee.getFunctionType(),
      methodCallOp.getOperands()
    );

    return success();
  }
};

}

void MonomorphizePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // phase 1: apply rewrite patterns
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
}

std::unique_ptr<Pass> createMonomorphizePass() {
  return std::make_unique<MonomorphizePass>();
}

} // end mlir::trait
