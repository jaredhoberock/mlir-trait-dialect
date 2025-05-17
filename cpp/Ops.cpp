#include "Dialect.hpp"
#include "Monomorphization.hpp"
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


std::optional<std::string> ImplOp::cloneAndSubstituteMissingOptionalTraitMethodsIntoBody(OpBuilder& builder) {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return "not inside a module";

  // search in the module for the trait
  auto traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, getTraitAttr());
  if (!traitOp)
    return "could not find trait";

  // for each optional method in the trait,
  // if this ImplOp does not provide the method,
  // clone and monomorphize the default implementation into the trait
  for (auto method : traitOp.getOptionalMethods()) {
    if (!hasMethod(method.getSymName())) {
      // XXX is there a way to do this without creating this extra clone?
      auto methodClone = cloneAndSubstituteReceiverType(method, getReceiverType());
      if (!methodClone)
        return "method cloning failed";

      PatternRewriter::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&getBody().front());
      builder.clone(*methodClone);

      // we no longer need the temporary clone
      methodClone.erase();
    }
  }

  return std::nullopt;
}


void ImplOp::mangleMethodNamesAndMoveIntoParent(RewriterBase& rewriter) {
  PatternRewriter::InsertionGuard guard(rewriter);

  // collect all methods
  std::vector<func::FuncOp> methods = getMethods();

  // hoist methods into the parent op with mangled names
  for (auto method : methods) {
    // mangle the method name before moving it
    method.setSymName(mangleMethodName(
      getTrait(),
      getReceiverType(),
      method.getSymName()
    ));

    // move the method into the ImplOp's parent
    rewriter.moveOpBefore(method, *this);
  }
}


// XXX TODO this should use apply(ty, substitution)
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
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError() << "not contained in a module";

  // extract trait and method from the nested symbol reference
  auto methodRef = getMethodRef();
  if (methodRef.getNestedReferences().empty())
    return emitOpError() << "expected nested symbol reference with trait::method format";

  auto traitAttr = getTraitAttr();
  auto methodAttr = cast<FlatSymbolRefAttr>(methodRef.getNestedReferences().front());
  auto receiverTyAttr = getReceiverTypeAttr();

  // look up the TraitOp
  auto traitOp = dyn_cast_or_null<TraitOp>(SymbolTable::lookupSymbolIn(moduleOp, traitAttr));
  if (!traitOp)
    return emitOpError()
      << "cannot find trait '" << traitAttr << "'";

  // look up the method in the trait
  auto method = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(traitOp, methodAttr);
  if (!method) {
    return emitOpError() << "cannot find method '" << methodAttr << "' in trait '" << traitAttr << "'";
  }

  Type monoReceiverTy = receiverTyAttr.getValue();

  // if monoSelfTy is !trait.poly, verify that the trait appears in its constraints
  if (auto paramTy = dyn_cast<PolyType>(monoReceiverTy)) {
    if (!llvm::is_contained(paramTy.getTraits(), traitAttr))
      return emitOpError()
        << paramTy << " is not constrained by trait '" << traitAttr << "'";
  }

  // monomorphize the method's type using the concrete receiver type
  FunctionType monoFnTy = monomorphizeFunctionType(
      method.getFunctionType(), 
      monoReceiverTy);

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
        impl.getReceiverTypeAttr().getValue() == type) {
      return true;
    }
  }

  return false;
}

static LogicalResult resolvePolyType(Location loc,
                                     PolyType polyTy,
                                     Type monoTy,
                                     llvm::DenseMap<Type,Type> &substitution,
                                     ModuleOp module,
                                     StringRef what,
                                     unsigned position) {
  // If we've already substituted this poly index, ensure consistency
  if (auto it = substitution.find(polyTy);
      it != substitution.end()) {
    if (it->second != monoTy) {
      return mlir::emitError(loc)
             << "mismatched substitution for poly type " << polyTy.getUniqueId()
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
             << " for poly type " << polyTy.getUniqueId() << " in " << what << " " << position;
    }
  }

  substitution[polyTy] = monoTy;
  return success();
}

static LogicalResult checkPolymorphicFunctionCall(
    FunctionType polyFnTy,
    TypeRange operandTypes,
    TypeRange resultTypes,
    Location loc,
    ModuleOp moduleOp) {
  llvm::DenseMap<Type, Type> substitution;

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
      if (failed(resolvePolyType(loc, poly, arg, substitution, moduleOp, "operand", i)))
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
      if (auto it = substitution.find(poly); it != substitution.end()) {
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

DenseMap<Type, Type> FuncCallOp::buildMonomorphicSubstitution() {
  auto calleeAttr = getCalleeAttr();
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, calleeAttr);

  TypeRange polymorphicParameterTypes = callee.getFunctionType().getInputs();
  TypeRange argumentTypes = getOperandTypes();

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

  return substitution;
}

func::FuncOp FuncCallOp::cloneAndMonomorphizeCalleeAtInsertionPoint(
    OpBuilder &builder,
    StringRef monomorphName) {
  func::FuncOp polymorph = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  llvm::DenseMap<Type, Type> substitution = buildMonomorphicSubstitution();

  if (polymorph.isExternal()) {
    polymorph.emitError("cannot monomorphize external function");
    return nullptr;
  }

  if (!isPolymorph(polymorph)) {
    polymorph.emitError("cannot monomorphize function that is not polymorphic");
    return nullptr;
  }

  MLIRContext *ctx = (*this)->getContext();

  // clone the polymorph using a temporary builder
  OpBuilder tempBuilder(ctx);
  func::FuncOp tempMonomorph = cast<func::FuncOp>(tempBuilder.clone(*polymorph));
  tempMonomorph.setSymName(monomorphName);

  if (failed(applySubstitution(tempMonomorph, substitution))) {
    tempMonomorph.erase();
    return nullptr;
  }

  // clone the temporary monomorph using the caller's builder
  func::FuncOp monomorph = cast<func::FuncOp>(builder.clone(*tempMonomorph));

  // we no longer need the temporary monomorph
  tempMonomorph.erase();

  return monomorph;
}

func::FuncOp FuncCallOp::getOrCreateMonomorphicCallee(OpBuilder& builder) {
  // lookup the callee
  auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCalleeAttr());
  if (!callee) {
    emitOpError("could not find callee");
    return nullptr;
  }

  // build the monomorphic substitution
  DenseMap<Type, Type> substitution = buildMonomorphicSubstitution();

  // get the name of the monomorphic callee
  std::string monomorphName = manglePolymorphicFunctionName(callee, substitution);

  // lookup the monomorphic callee
  func::FuncOp monomorph =
    SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, builder.getStringAttr(monomorphName));

  if (!monomorph) {
    // the monomorph doesn't exist yet; create it
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(callee);
    monomorph = cloneAndMonomorphizeCalleeAtInsertionPoint(builder, monomorphName);
    if (!monomorph) {
      emitOpError("monomorphization failed");
      return nullptr;
    }
  }

  return monomorph;
}

// inserts a func.call op at builder's insertion point to this trait.func.call op's
// monomorphized callee. If the callee's monomorph does not exist yet,
// the polymorphic callee will be cloned and monomorphized into the module
// if the intended edit fails, returns an error string
func::CallOp FuncCallOp::monomorphizeAtInsertionPoint(OpBuilder &builder) {
  func::FuncOp monomorph = getOrCreateMonomorphicCallee(builder);
  if (!monomorph) {
    emitOpError("monomorphization failed");
    return nullptr;
  }

  // insert a normal func.call op to the monomorphic callee
  return builder.create<func::CallOp>(
    getLoc(),
    monomorph.getSymName(),
    getResultTypes(),
    getOperands()
  );
}
