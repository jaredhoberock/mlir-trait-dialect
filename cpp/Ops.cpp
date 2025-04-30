#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
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

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify trait attribute exists
  auto traitAttr = getTraitAttr();
  if (!traitAttr)
    return emitOpError() << "requires a 'trait.trait' symbol reference attribute";

  // search in the enclosing operation's symbol table for the trait
  auto traitOp = symbolTable.lookupNearestSymbolFrom<TraitOp>(getOperation()->getParentOp(), traitAttr);
  if (!traitOp) {
    return emitOpError() << "cannot find trait '" << traitAttr << "'";
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
                             << "' (not found in trait '" << traitAttr << "')";
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
      return emitOpError() << "missing implementation for required method '" << name << "' of trait '" << traitAttr << "'";
    }
  }

  return success();
}

FunctionType monomorphizeFunctionType(FunctionType polyFnTy,
                                      Type monoSelfTy) {
  auto monomorphize = [&](Type type) -> Type {
    if (isa<SelfType>(type)) return monoSelfTy;
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
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError() << "not contained in a module";

  auto traitAttr = getTraitAttr();
  auto methodAttr = getMethodAttr();
  auto selfTyAttr = getSelfTypeAttr();

  // look up the TraitOp
  auto traitOp = dyn_cast_or_null<TraitOp>(SymbolTable::lookupSymbolIn(moduleOp, traitAttr));
  if (!traitOp)
    return emitOpError()
      << "cannot find trait '" << traitAttr << "'";

  // look up the method
  auto method = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(traitOp, methodAttr);
  if (!method) {
    return emitOpError() << "cannot find method '" << methodAttr << "' in trait '" << traitAttr << "'";
  }

  Type monoSelfTy = selfTyAttr.getValue();

  // if monoSelfTy is !trait.poly, verify that the trait appears in its constraints
  if (auto paramTy = dyn_cast<PolyType>(monoSelfTy)) {
    if (!llvm::is_contained(paramTy.getTraits(), traitAttr))
      return emitOpError()
        << paramTy << " is not constrained by trait '" << traitAttr << "'";
  }

  // monomorphize the method's type using the concrete self type
  FunctionType monoFnTy = monomorphizeFunctionType(
      method.getFunctionType(), 
      monoSelfTy);

  if (getOperands().getTypes() != monoFnTy.getInputs())
    return emitOpError() << "expected operand types " << monoFnTy.getInputs();

  if (getResultTypes() != monoFnTy.getResults())
    return emitOpError() << "expected result types" << monoFnTy.getResults();

  return success();
}

static bool typeHasImpl(Type type, FlatSymbolRefAttr traitRef, ModuleOp module) {
  auto traitOp = dyn_cast_or_null<TraitOp>(
      SymbolTable::lookupSymbolIn(module, traitRef));
  if (!traitOp)
    return false;

  for (ImplOp impl : module.getOps<ImplOp>()) {
    if (impl.getTraitAttr() == traitRef &&
        impl.getConcreteTypeAttr().getValue() == type) {
      return true;
    }
  }

  return false;
}

static LogicalResult resolvePolyType(Location loc,
                                     PolyType polyTy,
                                     Type monoTy,
                                     llvm::DenseMap<unsigned,Type> &substitutions,
                                     ModuleOp module,
                                     StringRef what,
                                     unsigned position) {
  auto id = polyTy.getUniqueId();

  // If we've already substituted this poly index, ensure consistency
  if (auto it = substitutions.find(id);
      it != substitutions.end()) {
    if (it->second != monoTy) {
      return mlir::emitError(loc)
             << "mismatched substitution for poly index " << id
             << " in " << what << " " << position
             << ": expected " << it->second << ", got " << monoTy;
    }
    return success();
  }

  // Ensure actual satisfies all trait constraints
  for (auto traitAttr : polyTy.getTraits()) {
    auto traitRef = cast<FlatSymbolRefAttr>(traitAttr);
    if (!typeHasImpl(monoTy, traitRef, module)) {
      return mlir::emitError(loc)
             << "type " << monoTy << " does not implement required trait " << traitRef
             << " for poly type " << id << " in " << what << " " << position;
    }
  }

  substitutions[id] = monoTy;
  return success();
}

static LogicalResult checkPolymorphicFunctionCall(
    FunctionType polyFnTy,
    TypeRange operandTypes,
    TypeRange resultTypes,
    Location loc,
    ModuleOp moduleOp) {
  llvm::DenseMap<unsigned, Type> substitutions;

  // check argument count
  auto paramTypes = polyFnTy.getInputs();
  if (operandTypes.size() != paramTypes.size())
    return emitError(loc) << "expected " << paramTypes.size() << " operands, but got "
                          << operandTypes.size();

  // check operand types
  for (auto [i, pair] : llvm::enumerate(llvm::zip(paramTypes, operandTypes))) {
    auto [param, arg] = pair;

    if (param == arg)
      continue;

    if (auto poly = dyn_cast<PolyType>(param)) {
      if (failed(resolvePolyType(loc, poly, arg, substitutions, moduleOp, "operand", i)))
        return failure();
    } else {
      return emitError(loc) << "operand " << i << " expected type " << param
                            << ", but got " << arg;
    }
  }

  // check result count
  auto expectedResults = polyFnTy.getResults();
  if (resultTypes.size() != expectedResults.size())
    return emitError(loc) << "expected " << expectedResults.size()
                          << " results, but got " << resultTypes.size();

  // check result types
  for (auto [i, pair] : llvm::enumerate(llvm::zip(expectedResults, resultTypes))) {
    auto [expected, actual] = pair;

    if (expected == actual)
      continue;

    if (auto poly = dyn_cast<PolyType>(expected)) {
      if (auto it = substitutions.find(poly.getUniqueId()); it != substitutions.end()) {
        if (it->second != actual) {
          return emitError(loc) << "result " << i << " expected " << it->second
                                << " (substituted for poly type " << poly.getUniqueId()
                                << "), but got " << actual;
        }
      } else {
        return emitError(loc) << "result " << i << " refers to poly type "
                              << poly.getUniqueId() << " with no substitution";
      }
    } else {
      return emitError(loc) << "result " << i << " expected type " << expected
                            << ", but got " << actual;
    }
  }

  return success();
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

  // check the types involved in a possibly polymorphic call
  return checkPolymorphicFunctionCall(funcOp.getFunctionType(), 
                                      getOperands().getTypes(), 
                                      getResultTypes(),
                                      getLoc(),
                                      moduleOp);
}
