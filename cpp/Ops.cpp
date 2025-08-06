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

  // check obligations
  for (auto& app : getObligations().getApplications()) {
    // each TraitApplicationAttr must use at least one of the trait's type parameters
    bool mentionsAny = llvm::any_of(uniqueParams, [&](PolyType param) {
      return app.mentionsType(param);
    });

    if (!mentionsAny)
      return emitOpError() << "'where' clause prerequisite " << app
                           << " must mention at least one type parameter";

    // must not refer to the current trait
    if (app.getTrait() == getSymNameAttr())
      return emitOpError() << "'where' clause prerequisite " << app
                           << " must not reference the current trait";
  }

  return success();
}


LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // verify obligations
  auto module = getOperation()->getParentOfType<ModuleOp>();
  return getObligations().verifyTraitApplications(module, [&](){ return emitOpError(); });
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

  // search through all ImplOps which use this TraitOp
  for (const auto& use : *uses) {
    auto impl = dyn_cast<ImplOp>(use.getUser());
    if (!impl) continue;

    SmallVector<Type> implTypes(impl.getTraitApplication().getTypeArgs());

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
    auto implTypeRange = impl.getTraitApplication().getTypeArgs();

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

SmallVector<TraitOp,4> TraitOp::getPrereqTraits() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("TraitOp::getPrereqTraits: not in a module");

  SmallVector<TraitOp,4> result;
  for (auto &app : getObligations().getApplications()) {
    TraitOp trait = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, app.getTrait());
    if (!trait)
      llvm_unreachable("TraitOp::getPrereqTraits: couldn't find prerequisite trait");
    result.push_back(trait);
  }
  return result;
}

LogicalResult TraitOp::verifyObligations(ImplOp impl, ArrayRef<ImplOp> obligations,
                                         llvm::function_ref<InFlightDiagnostic()> errFn) {
  // verify that impl's trait refers to this trait
  StringRef implTrait = impl.getTraitApplication().getTrait().getValue();
  if (implTrait != getSymName())
    return errFn() << "expected impl for @" << getSymName()
                   << ", but found impl for trait @" << implTrait;

  // verify that the received number of obligations matches our expectations
  size_t expectedNumTraitObligations = getObligations().getApplications().size();

  if (obligations.size() < expectedNumTraitObligations)
    return errFn() << "expected " << expectedNumTraitObligations << " for @"
                   << getSymName() << "'s 'where' clause, but found "
                   << obligations.size();

  // verify that each obligation implements the expected trait
  for (auto [app, ob] : llvm::zip(getObligations().getApplications(), obligations)) {
    StringRef expectedTrait = app.getTrait().getValue();
    StringRef obTrait = ob.getTraitApplication().getTrait().getValue();

    if (obTrait != expectedTrait)
      return errFn() << "expected impl for @" << expectedTrait
                     << ", but found impl for trait @" << obTrait;
  }

  return impl.verifyObligations(obligations.drop_front(expectedNumTraitObligations), errFn);
}


//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify trait application attribute exists
  auto traitApp = getTraitApplication();
  if (!traitApp)
    return emitOpError() << "requires a trait application attribute";

  // Get the trait
  auto traitOp = getTrait();
  if (!traitOp)
    return emitOpError() << "cannot find trait '" << traitApp.getTrait() << "'";

  // Check the trait's expected arity against typeArgs
  auto expectedArity = traitOp.getTypeParams().size();
  if (traitApp.getTypeArgs().size() != expectedArity)
    return emitOpError() << "trait '" << getTraitNameAttr() << "' expects " << expectedArity
                         << " type arguments, found " << traitApp.getTypeArgs().size();

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
                             << "' (not found in trait '" << getTraitNameAttr() << "')";
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
      return emitOpError() << "missing implementation for required method '" << name
                           << "' of trait '" << getTraitNameAttr() << "'";
    }
  }

  return success();
}

LogicalResult ImplOp::verifyObligations(ArrayRef<ImplOp> obligations,
                                        llvm::function_ref<InFlightDiagnostic()> errFn) {
  // verify that the received number of obligations matches our expectations
  size_t expectedNumObligations = getObligations().getApplications().size();

  if (obligations.size() != expectedNumObligations)
    return errFn() << "expected " << expectedNumObligations << "for @"
                   << getSymName() << "'s 'where' clause, but found "
                   << obligations.size();

  // verify that each obligation implements the expected trait
  for (auto [app, ob] : llvm::zip(getObligations().getApplications(), obligations)) {
    StringRef expectedTrait = app.getTrait().getValue();
    StringRef obTrait = ob.getTraitApplication().getTrait().getValue();

    if (obTrait != expectedTrait)
      return errFn() << "expected impl for @" << expectedTrait
                     << ", but found impl for trait @" << obTrait;
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
  return getTrait().buildSubstitutionFor(getTypeArgs());
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
      PatternRewriter::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&getBody().front());
      method = instantiatePolymorph(builder, traitMethod, methodName, buildSubstitution());
    }
  }

  return method;
}


static func::FuncOp cloneMethodAsFreeFuncWithLeadingSelfClaim(
    PatternRewriter& rewriter,
    func::FuncOp method,
    StringRef functionName,
    ClaimType selfClaimTy) {
  // clone and hoist method into the grandparent with a mangled name
  PatternRewriter::InsertionGuard guard(rewriter);

  // clone the method into the method's grandparent
  rewriter.setInsertionPointAfter(method->getParentOp());

  // clone the function
  auto funcOp = cast<func::FuncOp>(rewriter.clone(*method));

  // mutate the cloned op
  rewriter.modifyOpInPlace(funcOp, [&] {
    // set the name
    funcOp.setSymName(functionName);

    // prepend the claim parameter
    funcOp.insertArgument(/*idx=*/0, selfClaimTy,
                          /*argAttrs=*/mlir::DictionaryAttr(),
                          method.getLoc());
  });
  BlockArgument claimArg = funcOp.getArgument(0);

  // replace AssumeOps producing the claim type
  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    if (a.getResult().getType() == selfClaimTy) {
      rewriter.replaceAllUsesWith(a.getResult(), claimArg);
      toErase.push_back(a);
    }
  });

  // erase the AssumeOps
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

func::FuncOp ImplOp::getOrInstantiateFunctionFromMethod(PatternRewriter& rewriter, StringRef methodName) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return nullptr;

  // get the function name based on ImplOp symbol name
  auto functionName = getSymName().str() + "_" + methodName.str();

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      getOperation()->getParentOp(), FlatSymbolRefAttr::get(ctx, functionName));

  if (!funcOp) {
    // get the method inside the ImplOp
    func::FuncOp method = getOrInstantiateMethod(rewriter, methodName);
    
    // clone and hoist method into grandparent with mangled name
    funcOp = cloneMethodAsFreeFuncWithLeadingSelfClaim(
      rewriter,
      method,
      functionName,
      getSelfClaimType()
    );
  }

  return funcOp;
}

std::string ImplOp::generateSymName(TraitApplicationAttr traitApp,
                                    ObligationsAttr obligations) {
  std::string result;
  llvm::raw_string_ostream os(result);
  
  os << traitApp.getTrait().getValue() << "_impl";
  
  for (auto ty : traitApp.getTypeArgs()) {
    os << "_" << ty;
  }
  
  // Include where clause in symbol name if there are obligations
  if (!obligations.getApplications().empty()) {
    os << "_where";
    for (auto app : obligations.getApplications()) {
      os << "_" << app.getTrait().getValue();
      for (auto typeArg : app.getTypeArgs()) {
        os << "_" << typeArg;
      }
    }
  }
  
  return result;
}

ParseResult ImplOp::parse(OpAsmParser &p, OperationState &result) {
  // parse optional symbolic name: trait.impl @Sym
  StringAttr parsedSymName;
  (void)p.parseOptionalSymbolName(parsedSymName);

  // parse mandatory for
  if (p.parseKeyword("for"))
    return failure();
  
  // parse @TraitName[Types...]
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return failure();
  result.addAttribute("trait_application", traitApp);  
  
  // parse obligations
  ObligationsAttr obligations = dyn_cast<ObligationsAttr>(ObligationsAttr::parse(p, {}));
  if (!obligations)
    return failure();
  result.addAttribute("obligations", obligations);

  // sym_name: use parsed or synthesize from parameters
  StringAttr symNameAttr = parsedSymName
    ? parsedSymName
    : p.getBuilder().getStringAttr(generateSymName(traitApp, obligations));
  result.addAttribute("sym_name", symNameAttr);
  
  // Parse attributes and body region
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
    
  Region* bodyRegion = result.addRegion();
  if (p.parseRegion(*bodyRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  
  // Ensure the region has exactly one block (matching builder logic)
  if (bodyRegion->empty())
    bodyRegion->emplaceBlock();
    
  return success();
}

void ImplOp::print(OpAsmPrinter &printer) {
  // decide whether to print the symbolic name
  StringAttr symNameAttr = getSymNameAttr();
  std::string synthesized = generateSymName(getTraitApplication(), getObligations());
  bool printExplicitSymName = symNameAttr && symNameAttr.getValue() != synthesized;

  // Print: trait.impl [@SymName] for @TraitName [types...] obligations { ... }
  printer << " ";
  if (printExplicitSymName) {
    printer.printSymbolName(symNameAttr);
    printer << " ";
  }

  printer << "for ";
  getTraitApplication().print(printer);

  getObligations().print(printer);
  
  printer.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(), 
    /*elidedAttrs=*/{"sym_name", "trait_application", "obligations"}
  );
  printer << " ";
  printer.printRegion(getBody());
}


//===----------------------------------------------------------------------===//
// WitnessOp
//===----------------------------------------------------------------------===//

ParseResult WitnessOp::parse(OpAsmParser &p, OperationState& st) {
  // parse a TraitApplicationAttr: @Trait[Types...]
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return failure();

  // Result type inferred from TraitApplicationAttr
  st.addTypes(ClaimType::get(p.getContext(), traitApp));

  // parse optional trait obligations:
  // `where [ %o0: @A[Types...], %o1: @B[Types...], ... ]`
  SmallVector<OpAsmParser::UnresolvedOperand> obligations;
  SmallVector<Type> obligationTypes;

  if (succeeded(p.parseOptionalKeyword("where"))) {
    if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, [&] {
          OpAsmParser::UnresolvedOperand use;
          if (p.parseOperand(use) || p.parseColon())
            return failure();

          TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
          if (!traitApp)
            return failure();

          obligations.push_back(use);
          obligationTypes.push_back(ClaimType::get(p.getContext(), traitApp));
          return success();
        }))
    return failure();
  }

  // resolve operands
  if (p.resolveOperands(obligations, obligationTypes, p.getNameLoc(), st.operands))
    return failure();

  if (p.parseOptionalAttrDict(st.attributes)) return failure();

  return success();
}

void WitnessOp::print(OpAsmPrinter &p) {
  p << " ";
  getTraitApplication().print(p);

  // print trait obligations, if any
  if (!getTraitObligations().empty()) {
    p << " where [";
    p.increaseIndent();

    bool first = true;
    for (auto obligation : getTraitObligations()) {
      if (!first) {
        p << ",";
      }
      first = false;

      p.printNewline();
      p.printOperand(obligation);
      p << ": ";
      auto app = dyn_cast<ClaimType>(obligation.getType()).getTraitApplication();
      app.print(p);
    }

    p.decreaseIndent();
    p << "]";
  }
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  auto app = getTraitApplication();
  if (!app)
    return emitOpError() << "expected a TraitApplicationAttr";

  // look up the TraitOp
  auto traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, app.getTrait());
  if (!traitOp)
    return emitOpError() << "cannot find trait '" << app.getTrait() << "'";

  // find a matching ImplOp
  auto implOp = traitOp.getImpl(app.getTypeArgs());
  if (!implOp)
    return emitOpError() << "no matching trait.impl "
                         << app.getTrait() << " for " << app.getTypeArgs();

  // find a matching ImplOp for each trait obligation
  SmallVector<ImplOp> obligations;
  for (auto ob : getTraitObligations()) {
    auto app = cast<ClaimType>(ob.getType()).getTraitApplication();
    auto traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, app.getTrait());
    if (!traitOp)
      return emitOpError() << "cannot find trait '" << app.getTrait() << "'";

    auto implOp = traitOp.getImpl(app.getTypeArgs());
    if (!implOp)
      return emitOpError() << "no matching trait.impl "
                           << app.getTrait() << " for " << app.getTypeArgs();

    obligations.push_back(implOp);
  }

  // ask the trait to verify the impl's obligations
  auto errFn = [&] { return emitOpError(); };
  return traitOp.verifyObligations(implOp, obligations, errFn);
}


//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

ParseResult AssumeOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `@Trait[Types...]`
  TraitApplicationAttr app = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app) return failure();

  // result type is the claim of the trait application
  auto claimTy = ClaimType::get(p.getContext(), app);
  st.addTypes(claimTy);

  return success();
}

void AssumeOp::print(OpAsmPrinter &p) {
  p << " ";

  // print the witnessed trait application
  dyn_cast<ClaimType>(getResult().getType()).getTraitApplication().print(p);
}

LogicalResult AssumeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // verify line-of-sight between trait.assume op its enclosing function-like op so 
  // that we are able to replace uses of trait.assume with a function parameter
  Operation* isolatedAncestor = getOperation()->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  if (!isolatedAncestor)
    return emitOpError("must be within an IsolatedFromAbove region");

  // the isolated ancestor must be a FuncOp
  auto funcOp = dyn_cast<func::FuncOp>(isolatedAncestor);
  if (!funcOp)
    return emitOpError() << "must be within a 'func.func', found "
                         << isolatedAncestor->getName();

  // the function's immediate parent must be TraitOp or ImplOp
  TraitOp enclosingTrait = funcOp->getParentOfType<TraitOp>();
  ImplOp enclosingImpl = funcOp->getParentOfType<ImplOp>();
  
  if (!enclosingTrait && !enclosingImpl) {
    return emitOpError("must be within a 'trait.trait' or 'trait.impl' region");
  }
  
  FlatSymbolRefAttr assumedTraitAttr = getTraitAttr();
  ArrayRef<Type> assumedTypeArgs = getTypeArgs();
  
  if (enclosingTrait) {
    // In TraitOp: verify self-referential assumption
    StringAttr enclosingTraitName = enclosingTrait.getSymNameAttr();
    
    if (assumedTraitAttr.getValue() != enclosingTraitName.getValue()) {
      return emitOpError() << "assumed trait '" << assumedTraitAttr.getValue() 
                           << "' does not match enclosing trait '" 
                           << enclosingTraitName.getValue() << "'";
    }
    
    // Verify that type arguments match the enclosing trait's type parameters
    auto enclosingTypeParams = enclosingTrait.getTypeParams().getAsValueRange<TypeAttr>();
    if (!llvm::equal(assumedTypeArgs, enclosingTypeParams)) {
      return emitOpError() << "assumed claim type arguments must exactly match "
                           << "enclosing trait's type parameters";
    }
  } else if (enclosingImpl) {
    // In ImplOp: verify assumption matches the impl's trait and types
    FlatSymbolRefAttr implTraitAttr = enclosingImpl.getTraitNameAttr();
    
    if (assumedTraitAttr.getValue() != implTraitAttr.getValue()) {
      return emitOpError() << "assumed trait '" << assumedTraitAttr.getValue() 
                           << "' does not match impl's trait '" 
                           << implTraitAttr.getValue() << "'";
    }
    
    // Verify that type arguments match the impl's type arguments
    auto implTypeArgs = enclosingImpl.getTypeArgs();
    if (!llvm::equal(assumedTypeArgs, implTypeArgs)) {
      return emitOpError() << "assumed claim type arguments must exactly match "
                           << "enclosing impl's type arguments";
    }
  }
  
  return success();
}

FlatSymbolRefAttr AssumeOp::getTraitAttr() {
  return dyn_cast<ClaimType>(getResult().getType()).getTraitName();
}

ArrayRef<Type> AssumeOp::getTypeArgs() {
  return dyn_cast<ClaimType>(getResult().getType()).getTypeArgs();
}

TraitOp AssumeOp::getTrait() {
  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitOpError() << "not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(moduleOp, getTraitAttr());
}


//===----------------------------------------------------------------------===//
// MethodCallOp
//===----------------------------------------------------------------------===//

TraitOp MethodCallOp::getTrait() {
  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitOpError() << "method.call is not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(moduleOp, getTraitAttr());
}

LogicalResult MethodCallOp::verify() {
  // the claim's type must be an ClaimType
  ClaimType claim = dyn_cast_or_null<ClaimType>(getClaim().getType());
  if (!claim)
    return emitOpError() << "expected !trait.claim type, found " << getClaim().getType();

  // verify that the named trait matches the claim's trait
  auto expectedTraitAttr = getTraitAttr();
  auto foundTraitAttr = claim.getTraitName();
  if (expectedTraitAttr != foundTraitAttr)
    return emitOpError() << "expected claim for " << expectedTraitAttr << ", found " << foundTraitAttr;

  return success();
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

  // monomorphize the method's type using the claim's type arguments
  DenseMap<Type,Type> subst = traitOp.buildSubstitutionFor(getTraitApplication().getTypeArgs());
  FunctionType methodFnTy = dyn_cast_or_null<FunctionType>(applySubstitution(subst, method.getFunctionType()));
  if (!methodFnTy)
    return emitOpError() << "expected function type";

  return checkPolymorphicFunctionCall(methodFnTy,
                                      getArguments().getTypes(),
                                      getResultTypes(),
                                      getLoc(),
                                      moduleOp);
}

func::FuncOp MethodCallOp::getOrInstantiateCallee(PatternRewriter& rewriter) {
  return getTrait()
    .getOrInstantiateImpl(rewriter, getTraitApplication().getTypeArgs())
    .getOrInstantiateFunctionFromMethod(rewriter, getMethodName());
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
// ProjectOp
//===----------------------------------------------------------------------===//

ParseResult ProjectOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `%op: @Trait1[Types...] to @Trait2[Types...]`
  OpAsmParser::UnresolvedOperand src;
  if (p.parseOperand(src)) return failure();
  if (p.parseColon()) return failure();

  TraitApplicationAttr srcApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!srcApp) return failure();

  // operand type is the claim of the src trait application
  ClaimType srcTy = ClaimType::get(p.getContext(), srcApp);

  // resolve operand
  if (p.resolveOperand(src, srcTy, st.operands))
    return failure();

  if (p.parseKeyword("to"))
    return failure();

  TraitApplicationAttr resultApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!resultApp) return failure();

  // result type is the claim of the result application
  ClaimType resultTy = ClaimType::get(p.getContext(), resultApp);
  st.addTypes(resultTy);

  return success();
}

void ProjectOp::print(OpAsmPrinter& p) {
  // print `%op: %Trait1[Types...] to @Trait2[Types...]1

  p << " ";

  p.printOperand(getSource());
  p << ": ";
  dyn_cast<ClaimType>(getSource().getType()).getTraitApplication().print(p);

  p << " to ";
  dyn_cast<ClaimType>(getResult().getType()).getTraitApplication().print(p);
}

TraitOp ProjectOp::getSourceTrait() {
  auto claimTy = cast<ClaimType>(getSource().getType());
  auto module = getOperation()->getParentOfType<ModuleOp>();
  return SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, claimTy.getTraitName());
}

TraitOp ProjectOp::getProjectedTrait() {
  auto claimTy = cast<ClaimType>(getResult().getType());
  auto module = getOperation()->getParentOfType<ModuleOp>();
  return SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, claimTy.getTraitName());
}

LogicalResult ProjectOp::verifySymbolUses(SymbolTableCollection &/*symbolTable*/) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  auto errFn = [&]{ return emitOpError(); };

  // verify source claim
  auto srcClaim = cast<ClaimType>(getSource().getType());
  TraitApplicationAttr srcApp = srcClaim.getTraitApplication();
  if (failed(srcApp.verifyTraitApplication(module, errFn)))
    return failure();

  // verify destination claim
  auto dstClaim = cast<ClaimType>(getResult().getType());
  TraitApplicationAttr dstApp = dstClaim.getTraitApplication();
  if (failed(dstApp.verifyTraitApplication(module, errFn)))
    return failure();

  // get the source trait and check for a where clause
  TraitOp srcTrait = getSourceTrait();
  if (!srcTrait)
    return emitOpError() << "cannot resolve trait '"
                         << srcApp.getTrait() << "'";

  if (srcTrait.getObligations().getApplications().empty())
    return emitOpError() << "trait '" << srcTrait.getSymName()
                         << "' has no obligations; no projections allowed";

  // look up all candidate prerequisites that apply the dst trait
  auto candidates = srcTrait.getObligations().getApplicationsOf(dstApp.getTrait());
  if (candidates.empty())
    return emitOpError() << "trait '" << dstApp.getTrait()
                         << "' is not a prerequisite of trait '"
                         << srcApp.getTrait() << "'";

  // build substitution from the operand's type args
  auto subst = srcTrait.buildSubstitutionFor(srcApp.getTypeArgs());

  auto* ctx = getContext();

  // search for a candidate application, whose type args when applied
  // to this substitution, equal this type
  auto targetTuple = TupleType::get(ctx, dstApp.getTypeArgs());    // e.g. (i32, i32)

  for (auto candidate : candidates) {
    auto patternTuple = TupleType::get(ctx, candidate.getTypeArgs()); // e.g. (!S, !S)
    auto substTy = applySubstitution(subst, patternTuple);

    // note: if more than one candidate could be a match, that's fine, since
    // all matches name an identical trait application
    if (substTy == targetTuple)
      return success();
  }

  return emitOpError()
         << "projected trait application '" << dstApp
         << "' does not match substituted 'where' clause entry of '"
         << srcTrait.getSymName() << "'";
}


//===----------------------------------------------------------------------===//
// AllegeOp
//===----------------------------------------------------------------------===//

ParseResult AllegeOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `@Trait[Types...]`
  TraitApplicationAttr app = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app) return failure();

  // result type is the claim of the trait application
  auto claimTy = ClaimType::get(p.getContext(), app);
  st.addTypes(claimTy);

  return success();
}

void AllegeOp::print(OpAsmPrinter &p) {
  p << " ";

  // print the claimed trait application
  dyn_cast<ClaimType>(getResult().getType()).getTraitApplication().print(p);
}

LogicalResult AllegeOp::verify() {
  // type args must be concrete
  bool allConcrete = llvm::all_of(getTypeArgs(), [](Type ty) {
    return isConcrete(ty);
  });

  if (!allConcrete)
    return failure();

  return success();
}

TraitOp AllegeOp::getTrait() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    emitOpError() << "not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, getTraitAttr());
}

FlatSymbolRefAttr AllegeOp::getTraitAttr() {
  return getTraitApplication().getTrait();
}

LogicalResult AllegeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";
  return getTraitApplication().verifyTraitApplication(module, [this] { return emitOpError(); });
}

ArrayRef<Type> AllegeOp::getTypeArgs() {
  return dyn_cast<ClaimType>(getResult().getType()).getTypeArgs();
}

SmallVector<TraitApplicationAttr> AllegeOp::getPrereqTraitApplications() {
  TraitOp trait = getTrait();
  if (!trait)
    llvm_unreachable("AllegeOp::getPrereqTraitApplications: couldn't find TraitOp");

  SmallVector<TraitApplicationAttr> result;
  auto subst = trait.buildSubstitutionFor(getTypeArgs());

  for (auto prereqApp : trait.getObligations().getApplications()) {
    auto substApp = dyn_cast_or_null<TraitApplicationAttr>(applySubstitution(subst, prereqApp));
    if (!substApp)
      llvm_unreachable("AllegeOp::getPrereqTraitApplications: expected substituted trait application to be a TraitApplicationAttr");
    result.push_back(substApp);
  }

  return result;
}
