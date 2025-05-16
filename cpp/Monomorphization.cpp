#include "Monomorphization.hpp"
#include "Types.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::trait {

template<class MLIRType> bool containsType(TypeRange types) {
  auto hasType = [](Type t) { return isa<MLIRType>(t); };
  return llvm::any_of(types, hasType);
}

template<class MLIRType> bool containsType(ArrayRef<NamedAttribute> attrs) {
  for (const auto &attr : attrs) {
    if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
      if (isa<MLIRType>(typeAttr.getValue())) {
        return true;
      }
    }
  }
  return false;
}

template<class MLIRType> bool functionTypeContains(FunctionType ty) {
  return containsType<MLIRType>(ty.getInputs()) || containsType<MLIRType>(ty.getResults());
}

template<class MLIRType> bool mentionsType(Operation* op) {
  // check the operations's
  // * operand types,
  // * result types,
  // * and attribute types
  // for any mention of the mlir Type T of interest
  return containsType<MLIRType>(op->getOperandTypes()) ||
         containsType<MLIRType>(op->getResultTypes()) ||
         containsType<MLIRType>(op->getAttrs());
}

static bool containsSymbolicType(TypeRange types) {
  auto isSymbolic = [](Type t) { return isa<SymbolicTypeInterface>(t); };
  return llvm::any_of(types, isSymbolic);
}

static bool functionTypeContainsSymbolicType(FunctionType ty) {
  return containsSymbolicType(ty.getInputs()) || containsSymbolicType(ty.getResults());
}

bool isPolymorph(func::FuncOp fn) {
  return functionTypeContainsSymbolicType(fn.getFunctionType());
}

std::map<unsigned int, Type> buildMonomorphicSubstitutionForCall(TypeRange polymorphicParameterTypes,
                                                                 TypeRange argumentTypes) {
  if (polymorphicParameterTypes.size() != argumentTypes.size())
    llvm_unreachable("polymorphicParameterTypes.size() != argumentTypes.size()");

  std::map<unsigned int, Type> substitution;

  // for each unique PolyType in the polymorphic parameter types,
  // substitute the corresponding monomorphic argument type 
  for (auto [paramTy, argTy] : llvm::zip(polymorphicParameterTypes, argumentTypes)) {
    if (auto polyTy = dyn_cast<PolyType>(paramTy)) {
      unsigned int id = polyTy.getUniqueId();

      // check if substitution already exists
      auto it = substitution.find(id);
      if (it != substitution.end()) {
        // if it exists, ensure consistency
        if (it->second != argTy) {
          llvm_unreachable("Inconsistent substitution for PolyType");
        }
      } else {
        // if no substitution exists, create a new one
        substitution[id] = argTy;
      }
    }
  }
  
  return substitution;
}
                                                      
std::string manglePolymorphicFunctionName(func::FuncOp polymorph,
                                          const std::map<unsigned int, Type> &substitution) {
  std::string result = polymorph.getSymName().str();

  // append substituted concrete types to the mangled name
  for (auto [index, substitutedTy] : substitution) {
    llvm::raw_string_ostream os(result);
    os << "_";
    substitutedTy.print(os);
    os.flush();
  }

  return result;
}

template<class MLIRType>
struct ConvertAnyOpWithType : ConversionPattern {
  ConvertAnyOpWithType(TypeConverter &tc, MLIRContext *ctx)
    : ConversionPattern(tc, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (!mentionsType<MLIRType>(op)) {
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

      // check if the attribute is a TypeAttr with the MLIRType of interest
      if (auto typeAttr = dyn_cast<TypeAttr>(convertedAttr)) {
        if (MLIRType oldTy = dyn_cast<MLIRType>(typeAttr.getValue())) {
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

using ConvertAnyOpWithPolyType = ConvertAnyOpWithType<PolyType>;
using ConvertAnyOpWithSelfType = ConvertAnyOpWithType<SelfType>;

func::FuncOp monomorphizeFunction(func::FuncOp polymorph,
                                  const std::map<unsigned int, Type> &substitution) {
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

  // this type converter will convert each PolyType to its substituted concrete type
  TypeConverter typeConverter;
  typeConverter.addConversion([=](Type t) -> std::optional<Type> {
    if (PolyType polyTy = dyn_cast<PolyType>(t)) {
      if (auto it = substitution.find(polyTy.getUniqueId()); 
          it != substitution.end()) {
        return it->second;
      }
    }
    return t;
  });

  // mark illegal any operation which involves !trait.poly
  // ConvertAnyOpWithPolyType will monomorphize these operations
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp fn) {
    // a FuncOp is legal if it is not a polymorph
    return !isPolymorph(fn);
  });
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    // any random operation is legal if it does not mention PolyType
    return !mentionsType<PolyType>(op);
  });

  // build a set of rewrite patterns that simply apply the type converter to operand result types
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertAnyOpWithPolyType>(typeConverter, ctx);
  populateFunctionOpInterfaceTypeConversionPattern("func.func", patterns, typeConverter);
  if (failed(applyPartialConversion(monomorph, target, FrozenRewritePatternSet(std::move(patterns))))) {
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

  auto *ctx = method.getContext();

  // clone the method
  OpBuilder builder(ctx);
  func::FuncOp monomorph = cast<func::FuncOp>(builder.clone(*method));

  // this type converter will convert each placeholder SelfType to its concrete type
  TypeConverter typeConverter;
  typeConverter.addConversion([=](Type t) -> std::optional<Type> {
    if (isa<SelfType>(t)) {
      return concreteSelfType;
    }
    return t;
  });

  // mark illegal any operation which involves !trait.self
  // ConvertAnyOpWithSelfTypes will monomorphize these operations
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp fn) {
    // a FuncOp is legal if its type does not contain SelfType
    return !functionTypeContains<SelfType>(fn.getFunctionType());
  });
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    // any random operation is legal if does not mention SelfType
    return !mentionsType<SelfType>(op);
  });

  // build a set of rewrite patterns that simply apply the type converter to operand result types
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertAnyOpWithSelfType>(typeConverter, ctx);
  populateFunctionOpInterfaceTypeConversionPattern("func.func", patterns, typeConverter);

  if (failed(applyPartialConversion(monomorph, target, FrozenRewritePatternSet(std::move(patterns))))) {
    monomorph.erase();
    return nullptr;
  }

  return monomorph;
}

} // end mlir::trait
