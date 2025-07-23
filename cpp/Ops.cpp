#include "Dialect.hpp"
#include "Instantiation.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Verifier.h>
#include <iostream>
#include <optional>
#include <variant>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::trait;

static LogicalResult checkPolymorphicFunctionCall(
    FunctionType polyFnTy,
    TypeRange argumentTypes,
    TypeRange resultTypes,
    Location loc,
    ModuleOp moduleOp) {
  // check argument count
  auto paramTypes = polyFnTy.getInputs();
  if (argumentTypes.size() != paramTypes.size())
    return emitError(loc) << "expected " << paramTypes.size() << " arguments, but got "
                          << argumentTypes.size();

  // check result count
  auto expectedResults = polyFnTy.getResults();
  if (resultTypes.size() != expectedResults.size())
    return emitError(loc) << "expected " << expectedResults.size()
                          << " results, but got " << resultTypes.size();

  // create a FunctionType representing the caller's parameters and result
  FunctionType callerFnTy = FunctionType::get(moduleOp.getContext(), argumentTypes, resultTypes);
  DenseMap<Type, Type> subst;

  return unifyTypes(polyFnTy, callerFnTy, moduleOp, subst, [loc] {
    return mlir::emitError(loc);
  });
}

//===----------------------------------------------------------------------===//
// TraitOp
//===----------------------------------------------------------------------===//

LogicalResult TraitOp::verify() {
  auto typeParams = getTypeParams().getAsValueRange<TypeAttr>();

  // types must be unique PolyTypes
  DenseSet<PolyType> uniqueParams;
  for (Type ty : typeParams) {
    auto polyTy = dyn_cast<PolyType>(ty);
    if (!polyTy)
      return emitOpError() << "expected !trait.poly, found " << ty;
    if (!uniqueParams.insert(polyTy).second)
      return emitOpError() << "type parameters must be unique";
  }

  // there must be at least one type parameter
  if (uniqueParams.size() < 1)
    return emitOpError() << "requires at least one type parameter";

  for (Block &block : getBody()) {
    // check that all operations in the body are trait.method
    for (Operation &op : block) {
      if (!isa<MethodOp>(op))
        return emitOpError() << "body may only contain 'trait.method' operations";
    }
  }
  
  return success();
}

ImplOp TraitOp::getImpl(ArrayRef<Type> typeArgs) {
  MLIRContext* ctx = getContext();

  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;

  auto uses = mlir::SymbolTable::getSymbolUses(getOperation(), module);
  if (!uses)
    return nullptr;

  ImplOp symbolicImpl = nullptr;

  for (const auto& use : *uses) {
    auto impl = dyn_cast<ImplOp>(use.getUser());
    if (!impl) continue;

    SmallVector<Type> implTypes(impl.getTypeArgs().getAsValueRange<TypeAttr>());

    // first check the impl's type arguments for an exact match
    if (llvm::equal(implTypes, typeArgs)) {
      // as soon as we find an exact match for the typeArgs, we're done
      return impl;
    }

    // otherwise, check if the typeArgs can unify with the impl's type args,
    // indicating a "symbolic" match

    TupleType foundTy = TupleType::get(ctx, typeArgs);
    TupleType expectedTy = TupleType::get(ctx, implTypes);

    if (succeeded(unifyTypes(expectedTy, foundTy, module))) {
      // if there is more than one symbolic match, that's ambiguous, and an error
      if (symbolicImpl) return nullptr;
      symbolicImpl = impl;
    }
  }

  return symbolicImpl;
}

ImplOp TraitOp::getOrInstantiateImpl(OpBuilder& builder, ArrayRef<Type> typeArgs) {
  if (auto impl = getImpl(typeArgs)) {
    auto implTypeRange = impl.getTypeArgs().getAsValueRange<TypeAttr>();

    // check if the impl's type arguments are identical to typeArgs
    if (llvm::equal(implTypeRange, typeArgs)) {
      return impl;
    }

    // impl must be polymorphic; instantiate
    assert(llvm::any_of(implTypeRange, [](Type ty) {
      return containsSymbolicType(ty);
    }));
    
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(impl);
    return instantiatePolymorphicImpl(builder, impl, typeArgs);
  }

  return nullptr;
}

DenseMap<Type,Type> TraitOp::buildSubstitutionFor(TypeRange typeArgs) {
  DenseMap<Type,Type> subst;
  auto params = getTypeParams().getAsValueRange<TypeAttr>();
  for (auto [from, to] : llvm::zip(params, typeArgs)) {
    subst[from] = to;
  }
  return subst;
}

//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify trait name attribute exists
  auto traitNameAttr = getTraitNameAttr();
  if (!traitNameAttr)
    return emitOpError() << "requires a 'trait.trait' symbol reference attribute";

  // Get the trait
  auto traitOp = getTrait();
  if (!traitOp)
    return emitOpError() << "cannot find trait'" << traitNameAttr << "'";

  // Check the trait's expected arity against typeArgs
  auto expectedArity = traitOp.getTypeParams().size();
  if (getTypeArgs().size() != expectedArity)
    return emitOpError() << "trait '" << traitNameAttr << "' expects " << expectedArity
                         << " type arguments, found " << getTypeArgs().size();

  // XXX we should unify the trait's type parameters and the impl's type arguments here

  // Collect method names from the trait
  llvm::SmallSet<StringRef, 8> requiredMethodNames = traitOp.getRequiredMethodNames();
  std::vector<MethodOp> optionalMethods = traitOp.getOptionalMethods();
  llvm::SmallSet<StringRef, 8> optionalMethodNames;
  for (auto method : optionalMethods) {
    optionalMethodNames.insert(method.getSymName());
  }

  // Verify that the body contains only trait.method ops
  llvm::SmallSet<StringRef, 8> definedMethods;
  for (Operation &op : getBody().front()) {
    if (auto implMethod = dyn_cast<MethodOp>(op)) {
      StringRef name = implMethod.getSymName();
      if (!requiredMethodNames.contains(name) && !optionalMethodNames.contains(name)) {
        return emitOpError() << "implements unknown method '" << name
                             << "' (not found in trait '" << traitNameAttr << "')";
      }
      if (!implMethod.hasBody()) {
        return emitOpError() << "method '" << name << "' must have body";
      }
      if (!definedMethods.insert(name).second) {
        return emitOpError() << "implements method '" << name << "' multiple times";
      }
    } else {
      return emitOpError() << "body may only contain 'trait.method' operations";
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

DenseMap<Type, Type> ImplOp::buildSubstitution() {
  SmallVector<Type> typeArgs(getTypeArgs().getAsValueRange<TypeAttr>());
  return getTrait().buildSubstitutionFor(typeArgs);
}

std::vector<MethodOp> ImplOp::getMethods() {
  std::vector<MethodOp> result;
  for (auto& op : getBody().front()) {
    if (auto method = dyn_cast<MethodOp>(op)) {
      result.push_back(method);
    }
  }
  return result;
}

MethodOp ImplOp::getMethod(StringRef name) {
  return lookupSymbol<MethodOp>(name);
}

bool ImplOp::hasMethod(StringRef name) {
  return getMethod(name) != nullptr;
}

MethodOp ImplOp::getOrInstantiateMethod(OpBuilder& builder, StringRef methodName) {
  // check that we've named a valid trait method
  if (!getTrait().hasMethod(methodName)) return nullptr;

  // check if the method already exists in the ImplOp
  MethodOp method = getMethod(methodName);
  if (!method) {
    // we need to instantiate the method from the default implementation in the trait
    auto traitMethod = getTrait().getOptionalMethod(methodName);
    if (traitMethod) {
      // get the mangled name of the instance
      SmallVector<Type> typeArgs(getTypeArgs().getAsValueRange<TypeAttr>());
      auto instanceName = mangleMethodName(getTraitName(), typeArgs, methodName);

      PatternRewriter::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&getBody().front());
      method = instantiatePolymorphicMethod(builder, traitMethod, instanceName, buildSubstitution());
    }
  }

  return method;
}

func::FuncOp ImplOp::getOrInstantiateFunctionFromMethod(OpBuilder& builder, StringRef methodName) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return nullptr;

  // get the mangled name of the function instantiated from the method
  SmallVector<Type> typeArgs(getTypeArgs().getAsValueRange<TypeAttr>());
  auto functionName = mangleMethodName(getTraitName(), typeArgs, methodName);

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      getOperation()->getParentOp(), FlatSymbolRefAttr::get(ctx, functionName));

  if (!funcOp) {
    // instantiate the method as a free function with a mangled name

    // get the method inside the ImplOp
    MethodOp method = getOrInstantiateMethod(builder, methodName);

    // clone the method as a func::FuncOp and hoist to module level
    funcOp = method.cloneAsFuncOpAndHoistIntoGrandparent(builder, functionName);
  }

  return funcOp;
}

//===----------------------------------------------------------------------===//
// WitnessOp
//===----------------------------------------------------------------------===//

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // look up the TraitOp
  auto traitOp = getTrait();
  if (!traitOp)
    return failure();

  // look up the ImplOp
  auto implOp = traitOp.getImpl(getTypeArgs());
  if (!implOp)
    return emitOpError() << "no matching trait.impl "
                         << getTraitAttr() << " for " << getTypeArgs();

  return success();
}

FlatSymbolRefAttr WitnessOp::getTraitAttr() {
  return dyn_cast<WitnessType>(getResult().getType()).getTrait();
}

ArrayRef<Type> WitnessOp::getTypeArgs() {
  return dyn_cast<WitnessType>(getResult().getType()).getTypeArgs();
}

TraitOp WitnessOp::getTrait() {
  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitOpError() << "witness is not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(moduleOp, getTraitAttr());
}

//===----------------------------------------------------------------------===//
// MethodCallOp
//===----------------------------------------------------------------------===//

LogicalResult MethodCallOp::verify() {
  // the witness's type must be WitnessType
  WitnessType witness = dyn_cast_or_null<WitnessType>(getWitness().getType());
  if (!witness)
    return emitOpError() << "expected '!trait.witness', found " << getWitness().getType();

  // verify that the named trait matches the witness's trait
  auto expectedTraitAttr = getTraitAttr();
  auto foundTraitAttr = witness.getTrait();
  if (expectedTraitAttr != foundTraitAttr)
    return emitOpError() << "expected witness for " << expectedTraitAttr << ", found " << foundTraitAttr;

  return success();
}

TraitOp MethodCallOp::getTrait() {
  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitOpError() << "method.call is not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(moduleOp, getTraitAttr());
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return emitOpError() << "not contained in a module";

  auto traitAttr = getTraitAttr();
  auto methodAttr = getMethodAttr();

  // look up the TraitOp
  auto traitOp = getTrait();
  if (!traitOp)
    return emitOpError() << "cannot find trait '" << traitAttr << "'";

  // look up the method in the trait
  auto method = traitOp.getMethod(methodAttr.getValue());
  if (!method)
    return emitOpError() << "cannot find method '" << methodAttr << "' in trait '" << traitAttr << "'";

  // check that method's function type matches what we expect
  if (method.getFunctionType() != getMethodFunctionType()) {
    return emitOpError() << "'" << methodAttr.getValue() << "''s type " << method.getFunctionType()
                         << " does not match expected type " << getMethodFunctionType();
  }

  // monomorphize the method's type using the witness's type arguments
  DenseMap<Type,Type> subst = traitOp.buildSubstitutionFor(getWitness().getType().getTypeArgs());
  FunctionType methodFnTy = dyn_cast_or_null<FunctionType>(applySubstitution(subst, method.getFunctionType()));
  if (!methodFnTy)
    return emitOpError() << "expected function type";

  return checkPolymorphicFunctionCall(methodFnTy,
                                      getArguments().getTypes(),
                                      getResultTypes(),
                                      getLoc(),
                                      moduleOp);
}

func::FuncOp MethodCallOp::getOrInstantiateCallee(OpBuilder& builder) {
  return getTrait()
    .getOrInstantiateImpl(builder, getWitness().getType().getTypeArgs())
    .getOrInstantiateFunctionFromMethod(builder, getMethodName());
}

//===----------------------------------------------------------------------===//
// FuncCallOp
//===----------------------------------------------------------------------===//

LogicalResult FuncCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
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
  DenseMap<Type, Type> result;

  FunctionType expectedTy = getCalleeFunctionType();
  FunctionType foundTy = FunctionType::get(getContext(), getOperandTypes(), getResultTypes());

  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (failed(unifyTypes(expectedTy, foundTy, moduleOp, result)))
    llvm_unreachable("FuncCallOp::buildSubstitution: callee types and operand types did not unify");

  return result;
}

std::string FuncCallOp::getNameOfCalleeInstance() {
  return mangleFunctionName(getCallee(), buildSubstitution());
}

func::FuncOp FuncCallOp::instantiateCalleeAtInsertionPoint(OpBuilder &builder) {
  // lookup the polymorphic callee
  func::FuncOp callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  auto instanceName = getNameOfCalleeInstance();
  DenseMap<Type, Type> substitution = buildSubstitution();

  return instantiatePolymorph(builder, callee, instanceName, substitution);
}

func::FuncOp FuncCallOp::getOrInstantiateCallee(OpBuilder &builder) {
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto method = cast<MethodOp>((*this)->getParentOp());

  // The operand number and types must match the method signature.
  const auto &results = method.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing method (@"
           << method.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match method result type ("
                         << results[i] << ")"
                         << " in method @" << method.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

ParseResult MethodOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void MethodOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void MethodOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

func::FuncOp MethodOp::cloneAsFuncOpAndHoistIntoGrandparent(OpBuilder& builder, StringRef funcName) {
  // Create func::FuncOp in the grandparent (module level)
  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointAfter((*this)->getParentOp());
  
  func::FuncOp funcOp = builder.create<func::FuncOp>(getLoc(), funcName, getFunctionType());
  
  // Copy attributes from the original method (except name and function type)
  for (NamedAttribute attr : (*this)->getAttrs()) {
    StringRef attrName = attr.getName();
    if (attrName != getSymNameAttrName() && attrName != getFunctionTypeAttrName()) {
      funcOp->setAttr(attrName, attr.getValue());
    }
  }
  
  // Clone the method body, converting trait.return to func.return
  if (hasBody()) {
    Block* methodBlock = &getBody().front();
    Block* funcBlock = funcOp.addEntryBlock();
    
    IRMapping mapping;
    // Map the arguments
    for (auto [methodArg, funcArg] : llvm::zip(methodBlock->getArguments(), funcBlock->getArguments())) {
      mapping.map(methodArg, funcArg);
    }
    
    builder.setInsertionPointToEnd(funcBlock);
    
    // Clone all operations except the terminator
    // XXX if any used value is owned by the method's parent,
    //     we also need to clone its defining op
    for (Operation& op : methodBlock->without_terminator()) {
      builder.clone(op, mapping);
    }
    
    // Get the original terminator and create corresponding func.return
    auto traitReturn = cast<ReturnOp>(methodBlock->getTerminator());
    SmallVector<Value> mappedOperands;
    for (Value operand : traitReturn.getOperands()) {
      mappedOperands.push_back(mapping.lookupOrDefault(operand));
    }
    builder.create<func::ReturnOp>(traitReturn.getLoc(), mappedOperands);
  }
  
  return funcOp;
}
