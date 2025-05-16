#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Monomorphization.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::trait {

static std::string mangleMethodName(
    StringRef traitName,
    Type concreteSelfType,
    StringRef methodName) 
{
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "__trait_" << traitName;
  os << "_impl_";

  concreteSelfType.print(os);

  os << "_" << methodName; // e.g., "eq"

  return os.str();
}

struct ImplOpLowering : public OpConversionPattern<ImplOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ImplOp implOp, 
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {

    auto module = implOp->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(implOp, "not inside a module");

    // search in the module for the trait
    auto traitOp = SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, implOp.getTraitAttr());
    if (!traitOp)
      return implOp.emitOpError("could not find trait");

    // for each optional method in the trait,
    // if that impl does not provide this method,
    // clone and monomorphize the default implementation into the trait
    for (auto method : traitOp.getOptionalMethods()) {
      if (!implOp.hasMethod(method.getSymName())) {
        // XXX is there a way to do this without creating this extra clone?
        auto orphanMonomorph = cloneAndMonomorphizeSelfType(method, implOp.getConcreteType());
        if (!orphanMonomorph)
          return implOp.emitOpError("monomorphization failed");

        rewriter.setInsertionPointToEnd(&implOp.getBody().front());
        rewriter.clone(*orphanMonomorph);
        orphanMonomorph.erase();
      }
    }

    // collect all methods in the ImplOp
    std::vector<func::FuncOp> methods = implOp.getMethods();

    // hoist methods into the parent op with mangled names
    for (auto method : methods) {
      rewriter.moveOpBefore(method, implOp);
      method.setSymName(mangleMethodName(
        implOp.getTrait(),
        implOp.getConcreteType(),
        method.getSymName()
      ));
    }

    // after all methods are lowered, erase the ImplOp itself
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
    Type selfType = op.getSelfType();

    // if self type is still polymorphic, don't lower yet
    if (isa<PolyType>(selfType)) {
      return rewriter.notifyMatchFailure(op, "self type is still polymorphic");
    }

    // mangle the callee name
    auto calleeName = mangleMethodName(
      op.getTrait(),
      selfType,
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
    // collect all polymorphic func.func ops
    std::set<func::FuncOp> polymorphs;
    module.walk([&](func::FuncOp func) {
      if (isPolymorph(func)) {
        polymorphs.insert(func);
      }
    });

    // collect all trait.func.call ops
    // XXX consider whether traversing the users of each polymorph
    //     is a better way to collect these
    // XXX we need to do the same thing with trait.method.call ops
    SmallVector<FuncCallOp> calls;
    module.walk([&](FuncCallOp call) {
      calls.push_back(call);
    });

    // collect all needed monomorphs
    std::map<std::string, func::FuncOp> monomorphs;

    // process each call
    for (FuncCallOp call : calls) {
      auto calleeAttr = call.getCalleeAttr();
      auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, calleeAttr);

      if (!callee)
        return call.emitOpError("could not find callee");

      // check if callee is polymorphic
      if (!isPolymorph(callee)) {
        // callee is not polymorphic, just replace trait.func.call with func.call
        rewriter.setInsertionPoint(call);
        rewriter.replaceOpWithNewOp<func::CallOp>(
          call,
          calleeAttr,
          call.getResultTypes(),
          call.getOperands()
        );
        continue;
      }

      // callee is polymorphic
      polymorphs.insert(callee);

      // build the monomorphic substitution
      std::map<unsigned int, Type> substitution = call.buildMonomorphicSubstitution();

      // get the name of the monomorphic callee
      std::string monomorphName = manglePolymorphicFunctionName(callee, substitution);
      func::FuncOp monomorph;

      // find the monomorph if it already exists; create it if it doesn't
      if (auto it = monomorphs.find(monomorphName); it != monomorphs.end()) {
        monomorph = it->second;
      } else {
        // the monomorph doesn't exist yet; create it
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(callee);

        func::FuncOp monomorph =
          call.cloneAndMonomorphizeCalleeAtInsertionPoint(rewriter, monomorphName);
        if (!monomorph)
          return call.emitOpError("monomorphization failed");

        monomorphs[monomorphName] = monomorph;
      }

      // replace trait.func.call with func.call to monomorph
      rewriter.setInsertionPoint(call);
      rewriter.replaceOpWithNewOp<func::CallOp>(
          call,
          monomorphName,
          call.getResultTypes(),
          call.getOperands()
      );
    }

    // after all trait.func.call ops are replaced, it should be safe to erase polymorphs
    // XXX TODO in general, there can be other users of the polymorphs
    //          somehow we need to monomorphize other possible uses
    for (func::FuncOp polymorph : polymorphs) {
      rewriter.eraseOp(polymorph);
    }

    return success();
  }
};

void populateTraitToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  patterns.add<
    ImplOpLowering,
    MethodCallOpLowering,
    MonomorphizeModule,
    TraitOpLowering
  >(typeConverter, patterns.getContext());
}

}
