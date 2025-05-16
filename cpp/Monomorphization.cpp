#include "Monomorphization.hpp"
#include "Types.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::trait {

static Type apply(Type ty, const DenseMap<Type, Type> &substitution) {
  AttrTypeReplacer replacer;

  replacer.addReplacement([&](Type type) -> std::optional<Type> {
    auto it = substitution.find(type);
    if (auto it = substitution.find(type);
        it != substitution.end()) {
      return it->second;
    }
    return std::nullopt;
  });

  return replacer.replace(ty);
}

static bool containsSymbolicType(TypeRange types) {
  auto isSymbolic = [](Type t) { return isa<SymbolicTypeInterface>(t); };
  return llvm::any_of(types, isSymbolic);
}

static bool containsSymbolicType(ArrayRef<NamedAttribute> attrs) {
  for (const auto &attr : attrs) {
    if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
      if (isa<SymbolicTypeInterface>(typeAttr.getValue())) {
        return true;
      }
    }
  }
  return false;
}

// XXX TODO instead of mentionsSymbolicType,
//          it would probably be better simply to
//          apply some substitution to the op's operand types, result types, and attribute types
//          and check if there's any change
//
//          we could name this operationNeedsSubstitution(op, substitution)
static bool mentionsSymbolicType(Operation* op) {
  // check the operation's
  // * operand types,
  // * result types,
  // * and attribute types
  // for any mention of a SymbolicTypeInterface type
  return containsSymbolicType(op->getOperandTypes()) ||
         containsSymbolicType(op->getResultTypes()) ||
         containsSymbolicType(op->getAttrs());
}

static bool functionTypeContainsSymbolicType(FunctionType ty) {
  return containsSymbolicType(ty.getInputs()) || containsSymbolicType(ty.getResults());
}

bool isPolymorph(func::FuncOp fn) {
  return functionTypeContainsSymbolicType(fn.getFunctionType());
}
                                                      
std::string manglePolymorphicFunctionName(func::FuncOp polymorph,
                                          const DenseMap<Type, Type> &substitution) {
  std::string result = polymorph.getSymName().str();

  // append substituted concrete types to the mangled name
  for (auto [_, substitutedTy] : substitution) {
    llvm::raw_string_ostream os(result);
    os << "_";
    substitutedTy.print(os);
    os.flush();
  }

  return result;
}

// XXX we should pass the substitution to this conversion pattern
//     and simply check if op needs it
//     if it doesn't, we return failure
struct ConvertAnyOpWithSymbolicType : ConversionPattern {
  ConvertAnyOpWithSymbolicType(TypeConverter &tc, MLIRContext *ctx)
    : ConversionPattern(tc, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (!mentionsSymbolicType(op)) {
      return failure();
    }

    auto *tc = getTypeConverter();

    // Convert result types
    SmallVector<Type> newResults;
    for (Type t : op->getResultTypes()) {
      Type converted = tc->convertType(t);
      if (!converted) {
        return failure();
      }
      newResults.push_back(converted);
    }

    // Convert type attributes
    SmallVector<NamedAttribute> newAttrs;
    for (auto &attr : op->getAttrs()) {
      Attribute convertedAttr = attr.getValue();

      // check if the attribute is a TypeAttr with a SymbolicTypeInterface
      if (auto typeAttr = dyn_cast<TypeAttr>(convertedAttr)) {
        Type oldTy = typeAttr.getValue();
        if (isa<SymbolicTypeInterface>(oldTy)) {
          Type newTy = tc->convertType(oldTy);
          if (!newTy) {
            return failure();
          }

          convertedAttr = TypeAttr::get(newTy);
        }
      }

      newAttrs.push_back(NamedAttribute(attr.getName(), convertedAttr));
    }

    // Rebuild the op (using create with OperationState)
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(newResults);
    state.addAttributes(newAttrs);
    for (Region &region : op->getRegions())
      state.addRegion()->takeBody(region);

    // XXX seems like we would also need to recursively traverse any regions the op has

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

LogicalResult applySubstitution(func::FuncOp polymorph,
                                const DenseMap<Type, Type> &substitution) {
  // this type converter applies the substitution to the given type
  TypeConverter typeConverter;
  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    return apply(t, substitution);
  });

  MLIRContext* ctx = polymorph->getContext();

  // mark illegal any operation which involves !trait.poly
  // ConvertAnyOpWithPolyType will monomorphize these operations
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp fn) {
    // a FuncOp is legal if it is not a polymorph
    return !isPolymorph(fn);
  });
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    // any random operation is legal if it does not mention a SymbolicTypeInterface
    return !mentionsSymbolicType(op);
  });

  // build a set of rewrite patterns that simply apply the type converter to operand result types
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertAnyOpWithSymbolicType>(typeConverter, ctx);
  populateFunctionOpInterfaceTypeConversionPattern("func.func", patterns, typeConverter);

  // now do a partial conversion
  return applyPartialConversion(polymorph, target, FrozenRewritePatternSet(std::move(patterns)));
}

func::FuncOp monomorphizeFunction(func::FuncOp polymorph,
                                  const DenseMap<Type, Type> &substitution) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot monomorphize external function");
    return nullptr;
  }

  if (!isPolymorph(polymorph)) {
    polymorph.emitError("cannot monomorphize function that is not polymorphic");
    return nullptr;
  }

  ModuleOp module = polymorph->getParentOfType<ModuleOp>();

  // look for an existing monomorph
  std::string monomorphName = manglePolymorphicFunctionName(polymorph, substitution);
  if (auto existing = module.lookupSymbol<func::FuncOp>(monomorphName))
    return existing;

  auto *ctx = polymorph.getContext();

  // clone the polymorph
  OpBuilder builder(ctx);
  func::FuncOp monomorph = cast<func::FuncOp>(builder.clone(*polymorph));
  monomorph.setSymName(monomorphName);

  if (failed(applySubstitution(monomorph, substitution))) {
    monomorph.erase();
    return nullptr;
  }

  return monomorph;
}

func::FuncOp cloneAndMonomorphizeSelfType(func::FuncOp method,
                                          Type concreteSelfType) {
  if (method.isExternal()) {
    method.emitError("cannot monomorphize external function");
    return nullptr;
  }

  if (!isPolymorph(method)) {
    method.emitError("cannot monomorphize method that is not polymorphic");
    return nullptr;
  }

  auto *ctx = method.getContext();

  // clone the method
  OpBuilder builder(ctx);
  func::FuncOp monomorph = cast<func::FuncOp>(builder.clone(*method));

  // create a substitution mapping SelfType -> concreteSelfType
  llvm::DenseMap<Type,Type> substitution;
  substitution[SelfType::get(ctx)] = concreteSelfType;

  // apply the substitution
  if (failed(applySubstitution(monomorph, substitution))) {
    monomorph.erase();
    return nullptr;
  }

  return monomorph;
}

} // end mlir::trait
