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

static bool containsSymbolicType(Type ty) {
  // XXX TODO we actually need to traverse subelements of ty
  //          for this check to be correct
  return isa<SymbolicTypeInterface>(ty);
}

static bool containsSymbolicType(TypeRange types) {
  return llvm::any_of(types, [](Type t) {
    return containsSymbolicType(t);
  });
}

static bool containsSymbolicType(ArrayRef<NamedAttribute> attrs) {
  for (const auto &attr : attrs) {
    if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
      if (containsSymbolicType(typeAttr.getValue())) {
        return true;
      }
    }
  }
  return false;
}

static bool operationNeedsSubstitution(Operation* op, const DenseMap<Type,Type> &substitution) {
  // check the operation's
  // * operand types,
  // * result types,
  // * and attribute types
  // if any of them are transformed by the substitution,
  // then the op contains types that are mentioned in the domain of substitution

  // XXX TODO this search is probably inefficient since
  // apply() creates a new AttrTypeReplacer each time it is called

  for (auto ty : op->getOperandTypes()) {
    if (apply(ty, substitution) != ty)
      return true;
  }

  for (auto ty : op->getResultTypes()) {
    if (apply(ty, substitution) != ty)
      return true;
  }

  for (const auto &attr : op->getAttrs()) {
    if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
      Type ty = typeAttr.getValue();
      if (apply(ty, substitution) != ty)
        return true;
    }
  }

  return false;
}

static bool functionTypeContainsSymbolicType(FunctionType ty) {
  return containsSymbolicType(ty.getInputs()) || containsSymbolicType(ty.getResults());
}

bool isPolymorph(func::FuncOp fn) {
  return functionTypeContainsSymbolicType(fn.getFunctionType());
}

std::string mangleFunctionName(StringRef name,
                               const DenseMap<Type, Type> &substitution) {
  std::string result = name.str();

  // append substituted types to the name
  for (auto [_, substitutedTy] : substitution) {
    llvm::raw_string_ostream os(result);
    os << "_";
    substitutedTy.print(os);
    os.flush();
  }

  return result;
}
                                                      

std::string mangleMethodName(
    StringRef traitName,
    Type receiverType,
    StringRef methodName) 
{
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "__trait_" << traitName;
  os << "_impl_";

  receiverType.print(os);

  os << "_" << methodName; // e.g., "eq"

  return os.str();
}


struct ConvertAnyOpThatNeedsSubstitution : ConversionPattern {
  ConvertAnyOpThatNeedsSubstitution(
      TypeConverter &tc,
      MLIRContext *ctx,
      const llvm::DenseMap<Type,Type>& substitution)
    : ConversionPattern(tc, MatchAnyOpTypeTag(), 1, ctx),
      substitution(substitution) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (!operationNeedsSubstitution(op, substitution)) {
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

      // check if the attribute is a TypeAttr
      if (auto typeAttr = dyn_cast<TypeAttr>(convertedAttr)) {
        Type oldTy = typeAttr.getValue();

        // apply the substitution map
        Type substitutedTy = apply(oldTy, substitution);
        if (!substitutedTy)
          return failure();

        // if the substitution mapped oldTy to a different type,
        // convert the old type
        if (substitutedTy != oldTy) {
          Type convertedTy = tc->convertType(oldTy);
          if (!convertedTy)
            return failure();

          convertedAttr = TypeAttr::get(convertedTy);
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

  private:
    const llvm::DenseMap<Type,Type>& substitution;
};

LogicalResult applySubstitution(func::FuncOp polymorph,
                                const DenseMap<Type, Type> &substitution) {
  // this type converter applies the substitution to the given type
  TypeConverter typeConverter;
  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    return apply(t, substitution);
  });

  MLIRContext* ctx = polymorph->getContext();

  // mark illegal any operation which mentions a type in the domain of the substitution
  // ConvertAnyOpThatNeedsSubstitution will monomorphize these operations
  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    // any random operation is legal if it does not need the substitution
    return !operationNeedsSubstitution(op, substitution);
  });

  // build a set of rewrite patterns that converts ops that need the substitution
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertAnyOpThatNeedsSubstitution>(typeConverter, ctx, substitution);
  populateFunctionOpInterfaceTypeConversionPattern("func.func", patterns, typeConverter);

  // now do a partial conversion
  return applyPartialConversion(polymorph, target, FrozenRewritePatternSet(std::move(patterns)));
}

// XXX TODO this function shouldn't even exist, methods should
//          by monomorphized via monomorphizeFunction
func::FuncOp cloneAndSubstituteReceiverType(func::FuncOp method,
                                            Type receiverType) {
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

  // create a substitution mapping SelfType -> receiverType
  llvm::DenseMap<Type,Type> substitution;
  substitution[SelfType::get(ctx)] = receiverType;

  // apply the substitution
  if (failed(applySubstitution(monomorph, substitution))) {
    monomorph.erase();
    return nullptr;
  }

  return monomorph;
}

} // end mlir::trait
