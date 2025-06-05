#include "Dialect.hpp"
#include "Instantiation.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <iostream>
#include <optional>
#include <variant>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::trait;

LogicalResult TraitOp::verify() {
  for (Block &block : getBody()) {
    // check that all operations in the body are func.func
    for (Operation &op : block) {
      if (!isa<func::FuncOp>(op))
        return emitOpError() << "body may only contain 'func.func' operations";
    }
  }
  return success();
}

ImplOp TraitOp::getImpl(Type receiverTy) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    return nullptr;
  }

  auto uses = mlir::SymbolTable::getSymbolUses(getOperation(), module);
  if (!uses)
    return nullptr;

  ImplOp symbolicImpl = nullptr;

  for (const auto& use : *uses) {
    auto impl = dyn_cast<ImplOp>(use.getUser());
    if (!impl) continue;

    Type implReceiverTy = impl.getReceiverType();

    // first check the impl's receiver type for a direct match
    if (implReceiverTy == receiverTy) {
      // as soon as we find a direct match for the receiver type, we're done
      return impl;
    }

    // otherwise, check if the impl's receiver type is a symbolic matcher
    else if (auto symbolicTy = dyn_cast<SymbolicTypeUnificationInterface>(implReceiverTy)) {
      if (succeeded(symbolicTy.unifyWith(receiverTy, module))) {
        // if there is more than one symbolic match, that's ambiguous, and an error
        if (symbolicImpl) return nullptr;
        symbolicImpl = impl;
      }
    }
  }

  return symbolicImpl;
}

ImplOp TraitOp::getOrInstantiateImpl(OpBuilder& builder, Type receiverTy) {
  if (auto existingImpl = getImpl(receiverTy)) {
    // check if the impl's receiver type is identical to receiverTy
    if (existingImpl.getReceiverType() == receiverTy) {
      return existingImpl;
    }

    // existingImpl must be polymorphic; instantiate
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(existingImpl);
    return instantiatePolymorphicImpl(builder, existingImpl, receiverTy);
  }

  return nullptr;
}

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify trait attribute exists
  auto traitNameAttr = getTraitNameAttr();
  if (!traitNameAttr)
    return emitOpError() << "requires a 'trait.trait' symbol reference attribute";

  // Get the trait
  auto traitOp = getTrait();
  if (!traitOp) {
    return emitOpError() << "cannot find trait '" << traitNameAttr << "'";
  }

  // Collect method names from the trait
  llvm::SmallSet<StringRef, 8> requiredMethodNames = traitOp.getRequiredMethodNames();
  std::vector<func::FuncOp> optionalMethods = traitOp.getOptionalMethods();
  llvm::SmallSet<StringRef, 8> optionalMethodNames;
  for (auto f : optionalMethods) {
    optionalMethodNames.insert(f.getSymName());
  }

  // Verify that the body contains only func.func ops
  llvm::SmallSet<StringRef, 8> definedMethods;
  for (Operation &op : getBody().front()) {
    if (auto implMethod = dyn_cast<func::FuncOp>(op)) {
      StringRef name = implMethod.getSymName();
      if (!requiredMethodNames.contains(name) && !optionalMethodNames.contains(name)) {
        return emitOpError() << "implements unknown method '" << name
                             << "' (not found in trait '" << traitNameAttr << "')";
      }
      if (implMethod.isExternal()) {
        return emitOpError() << "method '" << name << "' must have body";
      }
      if (!definedMethods.insert(name).second) {
        return emitOpError() << "implements method '" << name << "' multiple times";
      }
    } else {
      return emitOpError() << "body may only contain 'func.func' operations";
    }
  }

  // Verify that no required methods are missing
  for (StringRef name : requiredMethodNames) {
    if (!definedMethods.contains(name)) {
      return emitOpError() << "missing implementation for required method '" << name << "' of trait '" << traitNameAttr << "'";
    }
  }

  return success();
}


TraitOp ImplOp::getTrait() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    emitOpError() << "impl is not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, getTraitNameAttr());
}


func::FuncOp ImplOp::getOrInstantiateMethod(OpBuilder& builder, StringRef methodName) {
  // check that we've named a valid trait method
  if (!getTrait().hasMethod(methodName)) return nullptr;

  // check if the method already exists in the ImplOp
  func::FuncOp method = getMethod(methodName);
  if (!method) {
    // we need to instantiate the method from the default implementation in the trait
    auto traitMethod = getTrait().getOptionalMethod(methodName);
    if (traitMethod) {
      auto instanceName = mangleMethodName(getTraitName(), getReceiverType(), methodName);
      // substitute receiver type for !trait.self
      DenseMap<Type,Type> substitution;
      substitution[SelfType::get(getContext())] = getReceiverType();

      PatternRewriter::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&getBody().front());
      method = instantiatePolymorph(builder, traitMethod, instanceName, substitution);
    }
  }

  return method;
}


func::FuncOp ImplOp::getOrInstantiateFunctionFromMethod(OpBuilder& builder, StringRef methodName) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return nullptr;

  // get the mangled name of the function instantiated from the method
  auto functionName = mangleMethodName(getTraitName(), getReceiverType(), methodName);

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      getOperation()->getParentOp(), FlatSymbolRefAttr::get(ctx, functionName));

  if (!funcOp) {
    // instantiate the method as a free function with a mangled name
    
    // get the method inside the ImplOp
    auto method = getOrInstantiateMethod(builder, methodName);

    // clone and hoist method into the parent with a mangled name
    PatternRewriter::InsertionGuard guard(builder);

    // clone the method into the ImplOp's parent
    builder.setInsertionPointAfter(*this);
    funcOp = cast<func::FuncOp>(builder.clone(*method));

    // set the method's name
    funcOp.setSymName(functionName);
  }

  return funcOp;
}


// XXX TODO this should use AttrTypeReplacer
FunctionType monomorphizeFunctionType(FunctionType polyFnTy,
                                      Type monoReceiverTy) {
  auto monomorphize = [&](Type type) -> Type {
    if (isa<SelfType>(type)) return monoReceiverTy;
    return type;
  };

  SmallVector<Type> monoInputTypes;
  for (Type ty : polyFnTy.getInputs())
    monoInputTypes.push_back(monomorphize(ty));

  SmallVector<Type> monoResultTypes;
  for (Type ty : polyFnTy.getResults())
    monoResultTypes.push_back(monomorphize(ty));

  return FunctionType::get(polyFnTy.getContext(), monoInputTypes, monoResultTypes);
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // get the various attributes
  auto traitAttr = getTraitAttr();
  auto methodAttr = getMethodAttr();
  auto receiverTyAttr = getReceiverTypeAttr();

  // look up the TraitOp
  auto traitOp = getTrait();
  if (!traitOp) {
    return emitOpError() << "cannot find trait '" << traitAttr << "'";
  }

  // look up the method in the trait
  auto method = traitOp.getMethod(methodAttr.getValue());
  if (!method) {
    return emitOpError() << "cannot find method '" << methodAttr << "' in trait '" << traitAttr << "'";
  }

  // check that method's function type matches what we expect
  if (method.getFunctionType() != getMethodFunctionType()) {
    return emitOpError() << "'" << methodAttr.getValue() << "''s type " << method.getFunctionType()
                         << " does not match expected type " << getMethodFunctionType();
  }

  Type receiverTy = receiverTyAttr.getValue();

  if (not isa<SymbolicTypeInterface>(receiverTy)) {
    // for a concrete type, check for an impl
    // XXX TODO if no concrete impl is found, check for a symbolic impl
    if (!traitOp.getImpl(receiverTy)) {
      return emitOpError()
        << receiverTy << " does not have a trait.impl for trait '" << traitAttr << "'";
    }
  }
  else if (auto paramTy = dyn_cast<PolyType>(receiverTy)) {
    // when receiverTy is !trait.poly, verify that the trait appears in its constraints
    if (!llvm::is_contained(paramTy.getTraits(), traitAttr))
      return emitOpError()
        << paramTy << " is not constrained by trait '" << traitAttr << "'";
  }

  // monomorphize the method's type using the concrete receiver type
  FunctionType monoFnTy = monomorphizeFunctionType(
      method.getFunctionType(), 
      receiverTy);

  if (getOperands().getTypes() != monoFnTy.getInputs())
    return emitOpError() << "expected operand types " << monoFnTy.getInputs();

  if (getResultTypes() != monoFnTy.getResults())
    return emitOpError() << "expected result types" << monoFnTy.getResults();

  return success();
}

TraitOp MethodCallOp::getTrait() {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp) {
    emitOpError() << "not contained in a module";
    return nullptr;
  }
  return dyn_cast_or_null<TraitOp>(SymbolTable::lookupSymbolIn(moduleOp, getTraitAttr()));
}

static DenseMap<Type,Type> buildSubstitutionForPossiblyPolymorphicCall(
    MLIRContext* ctx,
    TypeRange polymorphicParameterTypes,
    TypeRange argumentTypes,
    std::optional<Type> receiverType) {
  if (polymorphicParameterTypes.size() != argumentTypes.size())
    llvm_unreachable("polymorphicParameterTypes.size() != argumentTypes.size()");

  llvm::DenseMap<Type, Type> substitution;

  // for each unique PolyType in the polymorphic parameter types,
  // substitute the corresponding monomorphic argument type
  for (auto [paramTy, argTy] : llvm::zip(polymorphicParameterTypes, argumentTypes)) {
    if (isa<PolyType>(paramTy)) {
      // check if the substitution already exists
      auto it = substitution.find(paramTy);
      if (it != substitution.end()) {
        // if it exists, ensure consistency
        if (it->second != argTy) {
          llvm_unreachable("Inconsistent substitution for PolyType");
        } 
      } else {
        // if no substitution exists, create a new one
        substitution[paramTy] = argTy;
      }
    }
  }

  if (receiverType) {
    substitution[SelfType::get(ctx)] = *receiverType;
  }

  return substitution;
}

std::string MethodCallOp::getNameOfCalleeInstance() {
  return mangleMethodName(getTraitName(), getReceiverType(), getMethodName());
}

func::FuncOp MethodCallOp::getOrInstantiateCallee(OpBuilder& builder) {
  return getTrait()
    .getOrInstantiateImpl(builder, getReceiverType())
    .getOrInstantiateFunctionFromMethod(builder, getMethodName());
}

static LogicalResult checkPolymorphicFunctionCall(
    FunctionType polyFnTy,
    TypeRange operandTypes,
    TypeRange resultTypes,
    Location loc,
    ModuleOp moduleOp) {
  // check argument count
  auto paramTypes = polyFnTy.getInputs();
  if (operandTypes.size() != paramTypes.size())
    return emitError(loc) << "expected " << paramTypes.size() << " operands, but got "
                          << operandTypes.size();

  // check result count
  auto expectedResults = polyFnTy.getResults();
  if (resultTypes.size() != expectedResults.size())
    return emitError(loc) << "expected " << expectedResults.size()
                          << " results, but got " << resultTypes.size();

  // create a FunctionType representing the caller's parameters and result
  FunctionType callerFnTy = FunctionType::get(moduleOp.getContext(), operandTypes, resultTypes);
  llvm::DenseMap<Type, Type> substitution;

  return unifyTypes(loc, polyFnTy, callerFnTy, moduleOp, substitution);
}

LogicalResult FuncCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError() << "not contained in a module";

  auto calleeAttr = getCalleeAttr();
  if (!calleeAttr)
    return emitOpError() << "requires a 'callee' symbol reference attribute";

  auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, calleeAttr);
  if (!funcOp)
    return emitOpError() << "'" << calleeAttr.getValue()
                         << "' does not refer to a valid func.func";

  // check that funcOp's function type matches what we expect
  if (funcOp.getFunctionType() != getCalleeFunctionType()) {
    return emitOpError() << "'" << calleeAttr.getValue() << "''s type " << funcOp.getFunctionType()
                         << " does not match expected type " << getCalleeFunctionType();
  }

  // check the types involved in a possibly polymorphic call
  return checkPolymorphicFunctionCall(funcOp.getFunctionType(), 
                                      getOperands().getTypes(), 
                                      getResultTypes(),
                                      getLoc(),
                                      moduleOp);
}

DenseMap<Type, Type> FuncCallOp::buildSubstitution() {
  return buildSubstitutionForPossiblyPolymorphicCall(
      getContext(),
      getCalleeFunctionType().getInputs(),
      getOperandTypes(),
      std::nullopt);
}

std::string FuncCallOp::getNameOfCalleeInstance() {
  return mangleFunctionName(getCallee(), buildSubstitution());
}

func::FuncOp FuncCallOp::instantiateCalleeAtInsertionPoint(OpBuilder &builder) {
  // lookup the polymorphic callee
  func::FuncOp callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  auto instanceName = getNameOfCalleeInstance();
  llvm::DenseMap<Type, Type> substitution = buildSubstitution();

  return instantiatePolymorph(builder, callee, instanceName, substitution);
}

func::FuncOp FuncCallOp::getOrInstantiateCallee(OpBuilder& builder) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    emitOpError() << "not in a module";
    return nullptr;
  }

  // get the name of the instantiated callee
  std::string instanceName = getNameOfCalleeInstance();

  // look up the instance
  auto *symOp = SymbolTable::lookupSymbolIn(module, builder.getStringAttr(instanceName));
  auto instance = dyn_cast_or_null<func::FuncOp>(symOp);

  if (!instance) {
    // there's no instance yet, look up the polymorphic callee
    symOp = SymbolTable::lookupSymbolIn(module, getCalleeAttr());
    auto callee = dyn_cast_or_null<func::FuncOp>(symOp);
    if (!callee) {
      emitOpError() << "could not find callee " << getCalleeAttr();
      return nullptr;
    }

    // the instance doesn't exist yet; create it
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(callee);
    instance = instantiateCalleeAtInsertionPoint(builder);
    if (!instance) {
      emitOpError("instantiation failed");
      return nullptr;
    }
  }

  return instance;
}
