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

  // check requirements
  for (auto& app : getRequirements().getApplications()) {
    // each TraitApplicationAttr must use at least one of the trait's type parameters
    bool mentionsAny = llvm::any_of(uniqueParams, [&](PolyType param) {
      return app.mentionsType(param);
    });

    if (!mentionsAny)
      return emitOpError() << "'where' clause requirement " << app
                           << " must mention at least one type parameter";

    // must not refer to the current trait
    if (app.getTrait() == getSymNameAttr())
      return emitOpError() << "'where' clause requirement " << app
                           << " must not reference the current trait";
  }

  return success();
}


LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // verify obligations
  auto module = getOperation()->getParentOfType<ModuleOp>();
  return getRequirements().verifyTraitApplications(module, [&](){ return emitOpError(); });
}

DenseMap<Type,Type> TraitOp::buildSubstitutionFor(ClaimType claimTy) {
  auto module = getOperation()->getParentOfType<ModuleOp>();

  // unify our self claim with the given claim
  DenseMap<Type,Type> subst;
  if (failed(unifyTypes(getSelfClaim(), claimTy, module, subst)))
    llvm_unreachable("TraitOp::buildSubstitutionFor: unifyTypes failed");

  return subst;
}

SmallVector<TraitOp,4> TraitOp::getRequiredTraits() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("TraitOp::getPrereqTraits: not in a module");

  SmallVector<TraitOp,4> result;
  for (auto &app : getRequirements().getApplications()) {
    TraitOp trait = mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, app.getTrait());
    if (!trait)
      llvm_unreachable("TraitOp::getPrereqTraits: couldn't find required trait");
    result.push_back(trait);
  }
  return result;
}

LogicalResult TraitOp::verifyRequirements(ImplOp impl, ArrayRef<ImplOp> subproofs,
                                          llvm::function_ref<InFlightDiagnostic()> errFn) {
  // verify that impl's trait refers to this trait
  StringRef implTrait = impl.getSelfApplication().getTrait().getValue();
  if (implTrait != getSymName())
    return errFn() << "expected impl for @" << getSymName()
                   << ", but found impl for trait @" << implTrait;

  // verify that the received number of subproofs matches our expectations
  size_t numTraitReqs = getRequirements().getApplications().size();

  if (subproofs.size() < numTraitReqs)
    return errFn() << "expected " << numTraitReqs << " for @"
                   << getSymName() << "'s 'where' clause, but found "
                   << subproofs.size();

  // verify that each subproof impl implements the expected trait
  for (auto [app, proof] : llvm::zip(getRequirements().getApplications(), subproofs)) {
    StringRef expectedTrait = app.getTrait().getValue();
    StringRef proofTrait = proof.getSelfApplication().getTrait().getValue();

    if (proofTrait != expectedTrait)
      return errFn() << "expected impl for @" << expectedTrait
                     << ", but found impl for trait @" << proofTrait;
  }

  return impl.verifyAssumptions(subproofs.drop_front(numTraitReqs), errFn);
}

SmallVector<ClaimType> TraitOp::getRequirementsAsClaims() {
  SmallVector<ClaimType> result;
  for (TraitApplicationAttr app : getRequirements().getApplications()) {
    result.push_back(ClaimType::get(getContext(), app));
  }
  return result;
}


//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify self application attribute exists
  auto selfApp = getSelfApplication();
  if (!selfApp)
    return emitOpError() << "requires a self application TraitApplicationAttr";

  // Get the trait
  auto traitOp = getTrait();
  if (!traitOp)
    return emitOpError() << "cannot find trait '" << selfApp.getTrait() << "'";

  // Check the trait's expected arity against typeArgs
  auto expectedArity = traitOp.getTypeParams().size();
  if (selfApp.getTypeArgs().size() != expectedArity)
    return emitOpError() << "trait '" << getTraitNameAttr() << "' expects " << expectedArity
                         << " type arguments, found " << selfApp.getTypeArgs().size();

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

LogicalResult ImplOp::verifyAssumptions(ArrayRef<ImplOp> subproofs,
                                        llvm::function_ref<InFlightDiagnostic()> errFn) {
  // verify that the received number of subproofs matches our expectations
  size_t expectedNumProofs = getAssumptions().getApplications().size();

  if (subproofs.size() != expectedNumProofs)
    return errFn() << "expected " << expectedNumProofs << "for @"
                   << getSymName() << "'s 'where' clause, but found "
                   << subproofs.size();

  // verify that each subproof implements the expected trait
  for (auto [app, proof] : llvm::zip(getAssumptions().getApplications(), subproofs)) {
    StringRef expectedTrait = app.getTrait().getValue();
    StringRef proofTrait = proof.getSelfApplication().getTrait().getValue();

    if (proofTrait != expectedTrait)
      return errFn() << "expected impl for @" << expectedTrait
                     << ", but found impl for trait @" << proofTrait;
  }

  return success();
}

bool ImplOp::isSelfProof() {
  // an ImplOp is self-proven if its TraitOp has no requirements and the ImplOp has no assumptions
  return getAssumptions().getApplications().empty() && getTrait().getRequirements().getApplications().empty();
}

LogicalResult ImplOp::verifyIsSelfProof(llvm::function_ref<InFlightDiagnostic()> err) {
  if (!isSelfProof())
    return err() << "impl " << getSymName()
                 << " has trait requirements or impl assumptions and must be proven with a trait.proof";
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

DenseMap<Type, Type> ImplOp::buildSubstitutionFor(ClaimType claim) {
  // unify three types:
  // 1. Trait's self claim
  // 2. Impl's self claim
  // 3. The target claim

  // unify the impl's self claim with the trait's self claim
  DenseMap<Type,Type> subst = getTrait().buildSubstitutionFor(getSelfClaim());

  // now unify the impl's self claim with the target claim
  if (failed(unifyTypes(getSelfClaim(), claim, getParentOp<ModuleOp>(), subst)))
    llvm_unreachable("ImplOp::buildSubstitutionFor: failed to unify claim");

  return subst;
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

      auto subst = getTrait().buildSubstitutionFor(getSelfClaim());
      method = instantiatePolymorph(builder, traitMethod, methodName, subst);
    }
  }

  return method;
}

static func::FuncOp instantiateMethodAsFreeFuncWithLeadingSelfProof(
    PatternRewriter& rewriter,
    ModuleOp module,
    func::FuncOp method,
    StringRef functionName,
    ClaimType selfProofTy,
    const DenseMap<Type,Type>& subst) {
  // instantiate the method into the grandparent with a mangled name
  PatternRewriter::InsertionGuard guard(rewriter);

  // clone the method into the method's grandparent
  rewriter.setInsertionPointAfter(method->getParentOp());

  // instantiate the function
  auto funcOp = instantiatePolymorph(rewriter, method, functionName, subst);

  // mutate the cloned op
  rewriter.modifyOpInPlace(funcOp, [&] {
    // prepend the claim parameter
    funcOp.insertArgument(/*idx=*/0, selfProofTy,
                          /*argAttrs=*/mlir::DictionaryAttr(),
                          method.getLoc());
  });
  BlockArgument claimArg = funcOp.getArgument(0);

  // replace AssumeOps proven by selfProofTy
  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    if (a.getClaim() == selfProofTy) {
      rewriter.replaceAllUsesWith(a.getResult(), claimArg);
      toErase.push_back(a);
    }
  });

  // erase the AssumeOps
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

func::FuncOp ImplOp::getOrInstantiateFreeFunctionFromMethod(
    PatternRewriter& rewriter,
    ClaimType proof,
    StringRef methodName) {
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
    
    // instantiate into grandparent with mangled name
    auto subst = buildSubstitutionFor(proof);
    funcOp = instantiateMethodAsFreeFuncWithLeadingSelfProof(
      rewriter,
      getParentOp(),
      method,
      functionName,
      proof,
      subst
    );
  }

  return funcOp;
}

std::string ImplOp::generateSymName(TraitApplicationAttr selfApp,
                                    ConstraintsAttr assumptions) {
  std::string result;
  llvm::raw_string_ostream os(result);
  
  os << selfApp.getTrait().getValue() << "_impl";
  
  for (auto ty : selfApp.getTypeArgs()) {
    os << "_" << ty;
  }
  
  // Include where clause in symbol name if there are assumptions
  if (!assumptions.getApplications().empty()) {
    os << "_where";
    for (auto app : assumptions.getApplications()) {
      os << "_" << app.getTrait().getValue();
      for (auto typeArg : app.getTypeArgs()) {
        os << "_" << typeArg;
      }
    }
  }
  
  return result;
}

SmallVector<ClaimType> ImplOp::getRequirementsAsClaims() {
  // build substitution
  DenseMap<Type,Type> subst = buildSubstitutionFor(getSelfClaim());

  // specialize each requirement: @Req[params...] to @Req[args...]
  SmallVector<ClaimType> result;
  for (ClaimType polyReq : getTrait().getRequirementsAsClaims()) {
    ClaimType req = dyn_cast<ClaimType>(applySubstitution(subst, polyReq));
    if (!req)
      llvm_unreachable("ImplOp::getRequirementApplications: expected ClaimType");
    result.push_back(req);
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
  TraitApplicationAttr selfApp = dyn_cast<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!selfApp)
    return failure();
  result.addAttribute("self_application", selfApp);  
  
  // parse assumptions
  ConstraintsAttr assumptions = dyn_cast<ConstraintsAttr>(ConstraintsAttr::parse(p, {}));
  if (!assumptions)
    return failure();
  result.addAttribute("assumptions", assumptions);

  // sym_name: use parsed or synthesize from parameters
  StringAttr symNameAttr = parsedSymName
    ? parsedSymName
    : p.getBuilder().getStringAttr(generateSymName(selfApp, assumptions));
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
  std::string synthesized = generateSymName(getSelfApplication(), getAssumptions());
  bool printExplicitSymName = symNameAttr && symNameAttr.getValue() != synthesized;

  // Print: trait.impl [@SymName] for @TraitName [types...] assumptions { ... }
  printer << " ";
  if (printExplicitSymName) {
    printer.printSymbolName(symNameAttr);
    printer << " ";
  }

  printer << "for ";
  getSelfApplication().print(printer);

  printer << " ";
  getAssumptions().print(printer);
  
  printer.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(), 
    /*elidedAttrs=*/{"sym_name", "self_application", "assumptions"}
  );
  printer << " ";
  printer.printRegion(getBody());
}


//===----------------------------------------------------------------------===//
// ProofOp
//===----------------------------------------------------------------------===//

LogicalResult ProofOp::verify() {
  // check that subproofs is a non-empty array
  if (getSubproofNames().empty())
    return emitOpError() << "'subproof_names' must be a non-empty array";

  // check that every name is a FlatSymbolRefAttr
  for (Attribute name : getSubproofNames()) {
    if (!isa<FlatSymbolRefAttr>(name)) {
      return emitOpError() << "'subproof_names' must contain only FlatSymbolRefAttr elements";
    }
  }

  return success();
}

LogicalResult ProofOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *op = getOperation();
  auto errFn = [&] { return emitOpError(); };

  // check that the named impl exists
  auto implOp = getImpl();
  if (!implOp)
    return emitOpError() << "cannot find impl '" << getImplNameAttr() << "'";

  // check that our trait application matches the impl's trait application
  if (implOp.getSelfApplication() != getTraitApplication())
    return emitOpError() << "trait application does not match impl's declared trait application";

  // check that the named trait exists
  auto traitOp = getTrait();
  if (!traitOp)
    return emitOpError() << "cannot find trait '" << getTraitApplication().getTrait() << "'";

  // validate and collect each prereq ImplOp
  SmallVector<ImplOp> subproofImpls;
  for (Attribute name : getSubproofNames()) {
    auto subproofRef = dyn_cast<FlatSymbolRefAttr>(name);
    if (!subproofRef)
      return emitOpError() << "expected FlatSymbolRefAttr in 'subproof_names'";

    auto subproof = symbolTable.lookupNearestSymbolFrom(op, subproofRef);
    if (!subproof)
      return emitOpError() << "cannot find subproof '" << subproofRef << "'";

    // prereq needs to refer to either a ProofOp or an ImplOp with no obligations
    if (auto proofOp = dyn_cast<ProofOp>(subproof)) {
      // a ProofOp is fine
      subproofImpls.push_back(proofOp.getImpl());
    } else if (auto implOp = dyn_cast<ImplOp>(subproof)) {
      // an ImplOp can be its own proof
      if (failed(implOp.verifyIsSelfProof(errFn)))
        return failure();
      subproofImpls.push_back(implOp);
    } else {
      // the symbol refers to something that isn't a proof
      return emitOpError()
             << "symbol " << subproofRef.getValue()
             << " must refer to either a trait.proof or a trait.impl with no requirements or assumptions (i.e., no 'where' clauses)";
    }
  }

  // ask the traitOp to verify that its requirements (and the implOp's assumptions)
  // are fulfilled by prereqImpls
  return traitOp.verifyRequirements(implOp, subproofImpls, errFn);
}

TraitOp ProofOp::getTrait() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(module, getTraitApplication().getTrait());
}

SmallVector<ClaimType> ProofOp::getSubproofClaims() {
  SmallVector<ClaimType> result;

  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return result;

  for (Attribute name : getSubproofNames()) {
    auto subproofRef = dyn_cast<FlatSymbolRefAttr>(name);
    if (!subproofRef)
      llvm::report_fatal_error("ProofOp::getSubproofTypes: expected FlatSymbolRefAttr");

    auto subproof = SymbolTable::lookupNearestSymbolFrom(module, subproofRef);
    if (!subproof)
      llvm::report_fatal_error("ProofOp::getSubproofTypes: couldn't find referenced proof");

    // get the trait application of the subproof
    TraitApplicationAttr subproofTraitApp;
    if (auto proofOp = dyn_cast<ProofOp>(subproof)) {
      subproofTraitApp = proofOp.getTraitApplication();
    } else if (auto implOp = dyn_cast<ImplOp>(subproof)) {
      subproofTraitApp = implOp.getSelfApplication();
    } else {
      llvm::report_fatal_error("ProofOp::getSubproofTypes: expected ProofOp or ImplOp");
    }

    ClaimType subproofTy = ClaimType::get(getContext(), subproofTraitApp, subproofRef);
    result.push_back(subproofTy);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// WitnessOp
//===----------------------------------------------------------------------===//

ParseResult WitnessOp::parse(OpAsmParser &p, OperationState& result) {
  // parse @Symbol
  FlatSymbolRefAttr proof;
  if (p.parseAttribute(proof, "proof", result.attributes))
    return failure();

  // parse `for`
  if (p.parseKeyword("for"))
    return failure();

  // parse @Trait[Types...]
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addAttribute("trait_application", traitApp);

  // construct the result type
  ClaimType claimTy = ClaimType::get(p.getContext(), traitApp, proof);
  result.addTypes(claimTy);

  // parse additional attributes
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void WitnessOp::print(OpAsmPrinter &p) {
  p << " " << getProof() << " for ";
  getTraitApplication().print(p);

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"proof", "trait_application"}
  );
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto errFn = [&] { return emitOpError(); };

  auto proofRef = getProofAttr();
  auto proof = symbolTable.lookupNearestSymbolFrom(op, proofRef);
  if (!proof)
    return emitOpError() << "cannot find proof '" << proofRef << "'";

  // XXX much of the logic below is duplicated with ProofOp::verifySymbolUses

  ImplOp implOp;

  // proof needs to refer to either a ProofOp or an ImplOp with no obligations
  if (auto proofOp = dyn_cast<ProofOp>(proof)) {
    implOp = proofOp.getImpl();
    if (!implOp)
      return emitOpError() << "proof " << proofRef << " refers to missing impl";
  } else if (auto leafImplOp = dyn_cast<ImplOp>(proof)) {
    // an ImplOp can be a self-proof
    if (failed(leafImplOp.verifyIsSelfProof(errFn)))
      return failure();
    implOp = leafImplOp;
  } else {
    // the symbol refers to something that isn't a proof
    return emitOpError()
           << "symbol " << proofRef.getValue()
           << " must refer to either a trait.proof or a trait.impl with no obligations";
  }

  // ensure the impl matches the claimed trait application
  if (implOp.getSelfApplication() != getTraitApplication())
    return emitOpError() << "trait application does not match impl's declared trait application";

  return success();
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
  
  auto assumedApp = getTraitApplication();
  
  if (enclosingTrait) {
    // In TraitOp: verify self-referential application
    if (assumedApp != enclosingTrait.getSelfApplication())
      return emitOpError() << "assumed trait application " << assumedApp
                           << " does not match enclosing trait's self application "
                           << enclosingTrait.getSelfApplication();
  } else if (enclosingImpl) {
    // In ImplOp: allow assuming either the self application *or* any assumption
    if (assumedApp == enclosingImpl.getSelfApplication())
      return success();

    // is it one of the impl's assumptions?
    for (auto a : enclosingImpl.getAssumptions().getApplications()) {
      if (assumedApp == a)
        return success();
    }
    
    return emitOpError() << "assumed trait application " << assumedApp
                         << " is neither the impl's self application "
                         << enclosingImpl.getSelfApplication()
                         << " nor one of its declared assumptions";
  }
  
  return success();
}

TraitOp AssumeOp::getTrait() {
  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitOpError() << "not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<TraitOp>(moduleOp, getTraitApplication().getTrait());
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
  DenseMap<Type,Type> subst = traitOp.buildSubstitutionFor(cast<ClaimType>(getClaim().getType()));
  FunctionType methodFnTy = dyn_cast_or_null<FunctionType>(applySubstitution(subst, method.getFunctionType()));
  if (!methodFnTy)
    return emitOpError() << "expected function type";

  return checkPolymorphicFunctionCall(methodFnTy,
                                      getArguments().getTypes(),
                                      getResultTypes(),
                                      getLoc(),
                                      moduleOp);
}

ImplOp MethodCallOp::getProvenImpl() {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  assert(claimTy.isProven());

  auto proofRef = claimTy.getProof();
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();

  ImplOp result;
  auto proofOp = SymbolTable::lookupNearestSymbolFrom<ProofOp>(module, proofRef);
  if (proofOp) {
    result = proofOp.getImpl();
  } else {
    result = SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, proofRef);
  }

  return result;
}

func::FuncOp MethodCallOp::getOrInstantiateCallee(PatternRewriter& rewriter) {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  return getProvenImpl()
    .getOrInstantiateFreeFunctionFromMethod(rewriter, claimTy, getMethodName());
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
  std::string result = getCallee().str();

  auto subst = buildSubstitution();

  // append substituted types to the callee's name
  // except for ClaimTypes, which are not necessary
  // for distinguishing instances because
  // they are erased after monomorphization
  llvm::raw_string_ostream os(result);
  for (auto [_, substitutedTy] : subst) {
    if (!isa<ClaimType>(substitutedTy)) {
      os << "_" << substitutedTy;
    }
  }
  os.flush();

  return result;
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

  if (srcTrait.getRequirements().getApplications().empty())
    return emitOpError() << "trait '" << srcTrait.getSymName()
                         << "' has no requirements (i.e., no 'where' clause); no projections allowed";

  // look up all candidate requirements that apply the dst trait
  auto candidates = srcTrait.getRequirements().getApplicationsOf(dstApp.getTrait());
  if (candidates.empty())
    return emitOpError() << "trait '" << dstApp.getTrait()
                         << "' is not a requirement of trait '"
                         << srcApp.getTrait() << "'";

  // build substitution from the source claim
  auto subst = srcTrait.buildSubstitutionFor(srcClaim);

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
  // type args must be monomorphic
  bool allMonomorphic = llvm::all_of(getTypeArgs(), [](Type ty) {
    return isMonomorphicType(ty);
  });

  if (!allMonomorphic)
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

SmallVector<TraitApplicationAttr> AllegeOp::getRequiredTraitApplications() {
  TraitOp trait = getTrait();
  if (!trait)
    llvm_unreachable("AllegeOp::getRequiredTraitApplications: couldn't find TraitOp");

  ClaimType claimTy = dyn_cast<ClaimType>(getResult().getType());
  if (!claimTy)
    llvm_unreachable("AllegeOp::getRequiredTraitApplications: expected !trait.claim type");

  SmallVector<TraitApplicationAttr> result;
  auto subst = trait.buildSubstitutionFor(claimTy);

  for (auto reqApp : trait.getRequirements().getApplications()) {
    auto substApp = dyn_cast_or_null<TraitApplicationAttr>(applySubstitution(subst, reqApp));
    if (!substApp)
      llvm_unreachable("AllegeOp::getRequiredTraitApplications: expected substituted trait application to be a TraitApplicationAttr");
    result.push_back(substApp);
  }

  return result;
}
