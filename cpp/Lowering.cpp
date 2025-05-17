#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Monomorphization.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::trait {

struct FuncCallOpLowering : public OpConversionPattern<FuncCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FuncCallOp callOp,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // replace with a func.call to the monomorphic callee
    rewriter.replaceOpWithNewOp<func::CallOp>(
      callOp,
      callOp.getNameOfMonomorphicCallee(),
      callOp.getResultTypes(),
      callOp.getOperands()
    );
    return success();
  }
};

struct ImplOpLowering : public OpConversionPattern<ImplOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ImplOp implOp, 
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type receiverType = implOp.getReceiverType();

    // if receiver type is still symbolic, don't lower yet
    if (isa<SymbolicTypeInterface>(receiverType)) {
      return rewriter.notifyMatchFailure(implOp, "receiver type is still symbolic");
    }

    // clone any optional methods into the ImplOp that it needs
    if (auto errMsg = implOp.cloneAndSubstituteMissingOptionalTraitMethodsIntoBody(rewriter))
      return rewriter.notifyMatchFailure(implOp, *errMsg);

    // hoist all methods into the parent op with mangled names
    implOp.mangleMethodNamesAndMoveIntoParent(rewriter);

    // finally, erase the ImplOp itself
    rewriter.eraseOp(implOp);

    return success();
  }
};

struct MethodCallOpLowering : public OpConversionPattern<MethodCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MethodCallOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type receiverType = op.getReceiverType();

    // if receiver type is still symbolic, don't lower yet
    if (isa<SymbolicTypeInterface>(receiverType)) {
      return rewriter.notifyMatchFailure(op, "receiver type is still symbolic");
    }

    // mangle the callee name
    auto calleeName = mangleMethodName(
      op.getTrait(),
      receiverType,
      op.getMethod()
    );

    rewriter.replaceOpWithNewOp<func::CallOp>(
      op,
      calleeName,
      op.getResultTypes(),
      op.getOperands()
    );

    return success();
  }
};

struct TraitOpLowering : public OpConversionPattern<TraitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TraitOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // all we need to do for trait.trait is erase it
    rewriter.eraseOp(op);
    return success();
  }
};

struct MonomorphizeModule : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModuleOp module,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // collect all trait.func.call ops
    // XXX consider whether traversing the users of each polymorph
    //     is a better way to collect these
    // XXX we need to do the same thing with trait.method.call ops
    SmallVector<FuncCallOp> calls;
    module.walk([&](FuncCallOp call) {
      calls.push_back(call);
    });

    // for each call, ensure that a monomorphic callee exists
    for (FuncCallOp call : calls) {
      if (!call.getOrCreateMonomorphicCallee(rewriter))
        return call.emitOpError("monomorphization failed");
    }

    // collect & erase all polymorphs
    std::set<func::FuncOp> polymorphs;
    module.walk([&](func::FuncOp func) {
      if (isPolymorph(func)) {
        polymorphs.insert(func);
      }
    });

    for (auto polymorph : polymorphs) {
      rewriter.eraseOp(polymorph);
    }

    return success();
  }
};

void populateTraitToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  patterns.add<
    FuncCallOpLowering,
    ImplOpLowering,
    MethodCallOpLowering,
    MonomorphizeModule,
    TraitOpLowering
  >(typeConverter, patterns.getContext());
}

}
