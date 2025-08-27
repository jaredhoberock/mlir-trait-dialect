#include "Instantiation.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <iostream>
#include <optional>
#include <variant>

#define GET_OP_CLASSES
#include "TraitOps.cpp.inc"

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

  return substituteWith(polyFnTy, callerFnTy, moduleOp, subst, [loc] {
    return mlir::emitError(loc);
  });
}

// this function generates a mangled name suffix (e.g. "_i32_i1_f64", etc.)
// based on the substitution of some polymorphic entity (e.g., ImplOp, FuncOp, etc.)
static std::string generateMangledNameSuffixFor(const DenseMap<Type,Type> &subst, ArrayRef<PolyType> typeParams) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (auto ty : typeParams) {
    os << "_" << applySubstitutionToFixedPoint(subst, ty);
  }
  os.flush();
  return result;
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
  for (auto& app : getRequirements()) {
    // each TraitApplicationAttr must use at least one of the trait's type parameters
    bool mentionsAny = llvm::any_of(uniqueParams, [&](PolyType param) {
      return app.mentionsType(param);
    });

    if (!mentionsAny)
      return emitOpError() << "'where' clause requirement " << app
                           << " must mention at least one type parameter";

    // must not refer to the current trait
    if (app.getTraitName() == getSymNameAttr())
      return emitOpError() << "'where' clause requirement " << app
                           << " must not reference the current trait";
  }

  return success();
}


LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // verify obligations
  return getRequirements().verifyTraitApplications(getParentOp<ModuleOp>(), [&](){ return emitOpError(); });
}

DenseMap<Type,Type> TraitOp::buildSubstitutionFor(ClaimType claimTy) {
  // substitute our self claim with the given claim
  DenseMap<Type,Type> subst;
  if (failed(substituteWith(getSelfClaim(), claimTy, getParentOp<ModuleOp>(), subst)))
    llvm_unreachable("TraitOp::buildSubstitutionFor: substituteWith failed");
  return normalizeSubstitution(subst);
}

SmallVector<TraitOp,4> TraitOp::getRequiredTraits() {
  ModuleOp module = getParentOp<ModuleOp>();
  if (!module)
    llvm_unreachable("TraitOp::getPrereqTraits: not in a module");

  SmallVector<TraitOp,4> result;
  for (auto &app : getRequirements()) {
    auto trait = app.getTraitOrAbort(module, "TraitOp::getPrereqTraits: couldn't find required trait");
    result.push_back(trait);
  }
  return result;
}

SmallVector<ClaimType> TraitOp::getRequirementsAsClaimsWith(const DenseMap<Type,Type>& subst) {
  SmallVector<ClaimType> result;
  for (TraitApplicationAttr app : getRequirements()) {
    ClaimType in = ClaimType::get(getContext(), app);
    ClaimType out = dyn_cast_or_null<ClaimType>(applySubstitution(subst, in));
    if (!out)
      llvm_unreachable("TraitOp::getRequirementsAsClaimsWith: expected ClaimType");
    result.push_back(out);
  }
  return result;
}

SmallVector<ImplOp> TraitOp::getImpls() {
  ModuleOp module = getParentOp<ModuleOp>();
  if (!module)
    return {};

  // traverse users of this trait
  auto uses = mlir::SymbolTable::getSymbolUses(*this, module);
  if (!uses)
    return {};

  SmallVector<ImplOp> result;
  for (const auto& use : *uses) {
    if (auto impl = dyn_cast<ImplOp>(use.getUser())) {
      if (impl.getTrait() == *this)
        result.push_back(impl);
    }
  }

  return result;
}

SmallVector<ImplOp> TraitOp::getImplsFor(ClaimType claim) {
  SmallVector<ImplOp> result;

  ModuleOp module = getParentOp<ModuleOp>();
  if (!module)
    return result;

  for (auto impl : getImpls()) {
    auto implClaim = impl.getSelfClaim();

    // impl's claim is equivalent to the claim of interest
    // if each can substitute with the other
    bool eq = succeeded(substituteWith(implClaim, claim, module)) &&
              succeeded(substituteWith(claim, implClaim, module));
    if (eq) {
      result.push_back(impl);
    }
  }
  return result;
}


//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = getParentOp<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  // Verify self application attribute exists
  auto selfApp = getSelfApplication();
  if (!selfApp)
    return emitOpError() << "requires a self application TraitApplicationAttr";

  auto errFn = [&]{ return emitOpError(); };

  // Verify the self application
  if (failed(selfApp.verifyTraitApplication(module, errFn)))
    return failure();

  // Get the trait
  auto traitOp = getTrait();

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

bool ImplOp::isSelfProof() {
  // an ImplOp is self-proven if:
  // 1. it is monomorphic (no type parameters),
  // 2. its TraitOp has no requirements, and
  // 3. it has no assumptions
  return getTypeParams().empty() &&
         getAssumptions().empty() &&
         getTrait().getRequirements().empty();
}

LogicalResult ImplOp::verifyIsSelfProof(llvm::function_ref<InFlightDiagnostic()> err) {
  if (!isSelfProof()) {
    if (err) err() << "impl '@" << getSymName()
                   << "' is polymorphic (has type parameters) or has obligations (trait requirements or impl assumptions) and must be proven with a trait.proof";
    return failure();
  }
  return success();
}

TraitOp ImplOp::getTrait() {
  ModuleOp module = getParentOp<ModuleOp>();
  if (!module)
    llvm_unreachable("ImplOp::getTrait: not inside of a module");
  return getSelfApplication().getTraitOrAbort(module, "ImplOp::getTrait: couldn't find trait");
}

DenseMap<Type, Type> ImplOp::buildSubstitutionFor(ClaimType claim) {
  // unify three types:
  // 1. Trait's self claim
  // 2. Impl's self claim
  // 3. The target claim

  // unify the impl's self claim with the trait's self claim
  DenseMap<Type,Type> subst = getTrait().buildSubstitutionFor(getSelfClaim());

  // now substitute the impl's self claim with the target claim
  if (failed(substituteWith(getSelfClaim(), claim, getParentOp<ModuleOp>(), subst)))
    llvm_unreachable("ImplOp::buildSubstitutionFor: substituteWith failed");

  return normalizeSubstitution(subst);
}

SmallVector<PolyType, 4> ImplOp::getTypeParams() {
  auto selfClaim = getSelfClaim();
  auto subst = buildSubstitutionFor(selfClaim);

  // collect all the types where a PolyType could hide
  SmallVector<Type> allOurTypes;
  allOurTypes.push_back(selfClaim);
  for (ClaimType a : getAssumptionsAsClaimsWith(subst)) {
    allOurTypes.push_back(a);
  }

  // tuple the types
  TupleType tupled = TupleType::get(getContext(), allOurTypes);

  // get all the PolyTypes in the tuple
  return getPolyTypesIn(tupled);
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
  // note that this will instantiate invalid AssumeOps because their claims will include proofs
  // we'll fix the function by replacing AssumeOps below
  auto funcOp = instantiatePolymorph(rewriter, method, functionName, subst);

  // prepend the self proof as the first parameter of the function and
  // set visibility to private
  rewriter.modifyOpInPlace(funcOp, [&] {
    funcOp.insertArgument(/*idx=*/0, selfProofTy,
                          /*argAttrs=*/mlir::DictionaryAttr(),
                          method.getLoc());
    funcOp.setVisibility(SymbolTable::Visibility::Private);
  });
  BlockArgument selfProofArg = funcOp.getArgument(0);

  // replace all AssumeOps with projections of selfProofArg
  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(a);

    auto p = rewriter.create<ProjectOp>(
      a.getLoc(),
      a.getClaim(),
      selfProofArg
    );
    rewriter.replaceAllUsesWith(a.getResult(), p);
    toErase.push_back(a);
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

  // generate a free function name based on the ImplOp's mangled name for our proof
  auto functionName = generateMangledName(proof) + "_" + methodName.str();

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
    getParentOp(),
    FlatSymbolRefAttr::get(ctx, functionName)
  );

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
  
  os << selfApp.getTraitName().getValue() << "_impl";
  
  for (auto ty : selfApp.getTypeArgs()) {
    os << "_" << ty;
  }
  
  // Include where clause in symbol name if there are assumptions
  if (!assumptions.empty()) {
    os << "_where";
    for (auto app : assumptions) {
      os << "_" << app.getTraitName().getValue();
      for (auto typeArg : app.getTypeArgs()) {
        os << "_" << typeArg;
      }
    }
  }
  
  return result;
}

std::string ImplOp::generateMangledName(ClaimType claim) {
  DenseMap<Type,Type> subst = buildSubstitutionFor(claim);
  std::string result = getSymName().str();
  result += generateMangledNameSuffixFor(subst, getTypeParams());
  return result;
}

SmallVector<ClaimType> ImplOp::getAssumptionsAsClaimsWith(const DenseMap<Type,Type>& subst) {
  // specialize each assumption: @A[params...] to @A[Args...]
  SmallVector<ClaimType> result;
  for (auto polyApp : getAssumptions()) {
    ClaimType polyAssumption = ClaimType::get(getContext(), polyApp);
    ClaimType substAssumption = dyn_cast<ClaimType>(applySubstitution(subst, polyAssumption));
    if (!substAssumption)
      llvm_unreachable("ImplOp::getAssumptionsAsClaimsFor: expected ClaimType");
    result.push_back(substAssumption);
  }
  return result;
}

SmallVector<ClaimType> ImplOp::getObligationsAsClaimsWith(const DenseMap<Type,Type> &subst) {
  SmallVector<ClaimType> result = getTrait().getRequirementsAsClaimsWith(subst);
  result.append(getAssumptionsAsClaimsWith(subst));
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
  // check that the trait application is concrete
  auto app = getTraitApplication();
  for (auto ty : app.getTypeArgs()) {
    if (isPolymorphicType(ty))
      return emitOpError() << "'trait_application' must contain only concrete type arguments; found polymorphic type " << ty;
  }

  // check that every name is a FlatSymbolRefAttr
  for (Attribute name : getSubproofNames()) {
    if (!isa<FlatSymbolRefAttr>(name)) {
      return emitOpError() << "'subproof_names' must contain only FlatSymbolRefAttr elements";
    }
  }
  return success();
}

LogicalResult ProofOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = getParentOp<ModuleOp>();
  auto errFn = [&] { return emitOpError(); };

  // verify the claim
  if (failed(getProvenClaim().verifySymbolUses(module, errFn)))
    return success();

  // check that the named impl exists
  auto implOp = getImpl();
  if (!implOp)
    return emitOpError() << "cannot find impl '" << getImplNameAttr() << "'";

  // check that the impl's claim can be substitute with our proof
  // substituteWith will verify the details of the proof for us
  if (failed(substituteWith(implOp.getSelfClaim(), getProvenClaim(), module, errFn)))
    return failure();

  return success();
}

TraitOp ProofOp::getTrait() {
  auto module = getParentOp<ModuleOp>();
  if (!module)
    llvm_unreachable("ProofOp::getTrait: not inside a module");
  return getTraitApplication().getTraitOrAbort(module, "ProofOp::getTrait: couldn't find trait");
}

FailureOr<SmallVector<ClaimType>> ProofOp::verifyAndGetSubproofClaims(llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<ClaimType> result;

  ModuleOp module = getParentOp<ModuleOp>();
  if (!module) {
    if (err) err() << "not in a module";
    return failure();
  }

  for (Attribute name : getSubproofNames()) {
    auto subproofRef = dyn_cast<FlatSymbolRefAttr>(name);
    if (!subproofRef) {
      if (err) err() << "expected FlatSymbolRefAttr";
      return failure();
    }

    auto subproof = getProofOpOrSelfProofImplOp(module, subproofRef, err);
    if (failed(subproof)) {
      return failure();
    }

    // the subproof is guaranteed to be either ProofOp or ImplOp
    // get the trait application of the subproof
    TraitApplicationAttr subproofTraitApp;
    if (auto proofOp = dyn_cast<ProofOp>(*subproof)) {
      subproofTraitApp = proofOp.getTraitApplication();
    } else {
      subproofTraitApp = dyn_cast<ImplOp>(*subproof).getSelfApplication();
    }

    ClaimType subproofTy = ClaimType::get(getContext(), subproofTraitApp, subproofRef);
    result.push_back(subproofTy);
  }

  return result;
}

SmallVector<ClaimType> ProofOp::getImplAssumptionClaims() {
  SmallVector<ClaimType> result = getSubproofClaims();

  // drop the trait requirements from the subproofs
  size_t numTraitRequirements = getTrait().getRequirements().size();
  assert(numTraitRequirements <= result.size());

  result.erase(result.begin(), result.begin() + numTraitRequirements);
  return result;
}

FailureOr<Operation*> ProofOp::getProofOpOrSelfProofImplOp(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  Operation* symOp = SymbolTable::lookupNearestSymbolFrom(module, name);
  if (!symOp) {
    if (errFn) errFn() << "cannot find proof symbol '" << name << "'";
    return failure();
  }

  // if it's an ImplOp, it must be self-proving
  if (auto impl = dyn_cast<ImplOp>(symOp)) {
    if (failed(impl.verifyIsSelfProof(errFn))) return failure();
    return symOp;
  }

  // otherwise it must be a ProofOp
  auto proof = dyn_cast<ProofOp>(symOp);
  if (!proof) {
    if (errFn) errFn() << "proof symbol must refer to trait.proof or self-proving trait.impl";
    return failure();
  }

  return symOp;
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
  p << " " << getProofAttr() << " for ";
  getTraitApplication().print(p);

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"proof", "trait_application"}
  );
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitError() << "not inside a module";

  auto errFn = [&] { return emitOpError(); };

  // first verify the claim type
  if (failed(getProvenClaim().verifySymbolUses(module, errFn)))
    return failure();

  // look up the proof symbol 
  auto proofRef = getProofAttr();
  auto proof = SymbolTable::lookupNearestSymbolFrom(module, proofRef);
  if (!proof)
    return emitOpError() << "cannot find proof '" << proofRef << "'";

  // get the proof symbol's claim
  ClaimType proofSymClaim;

  // proof needs to refer to either a ProofOp or an ImplOp with no obligations
  if (auto proofOp = dyn_cast<ProofOp>(proof)) {
    proofSymClaim = proofOp.getProvenClaim();
  } else if (auto implOp = dyn_cast<ImplOp>(proof)) {
    proofSymClaim = implOp.getSelfClaim();
  } else {
    // the symbol refers to something that isn't a proof
    return emitOpError()
           << "symbol " << proofRef.getValue()
           << " must refer to either a trait.proof or a trait.impl with no obligations";
  }

  // we must be able to substitute the symbol's claim with our claim
  if (failed(substituteWith(proofSymClaim, getProvenClaim(), module, errFn)))
    return failure();

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
  getClaim().getTraitApplication().print(p);
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
    for (auto a : enclosingImpl.getAssumptions()) {
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
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("AssumeOp:getTrait: not inside of a module");
  return getTraitApplication().getTraitOrAbort(module, "AssumeOp::getTrait: couldn't find trait");
}


//===----------------------------------------------------------------------===//
// MethodCallOp
//===----------------------------------------------------------------------===//

LogicalResult MethodCallOp::verify() {
  // the claim's type must be an ClaimType
  ClaimType claim = dyn_cast_or_null<ClaimType>(getClaim().getType());
  if (!claim)
    return emitOpError() << "expected !trait.claim type, found " << getClaim().getType();

  // verify that the named trait matches the claim's trait
  auto expectedTraitAttr = getTraitAttr();
  auto foundTraitAttr = claim.getTraitApplication().getTraitName();
  if (expectedTraitAttr != foundTraitAttr)
    return emitOpError() << "expected claim for " << expectedTraitAttr << ", found " << foundTraitAttr;

  return success();
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not contained in a module";

  auto errFn = [&]{ return emitOpError(); };

  // verify the claim
  ClaimType claim = cast<ClaimType>(getClaim().getType());
  if (failed(claim.verifySymbolUses(module, errFn)))
    return failure();

  auto traitAttr = getTraitAttr();
  auto methodAttr = getMethodAttr();

  // look up the TraitOp
  auto traitOp = claim.getTraitApplication().getTraitOrAbort(module, "MethodCallOp::verifySymbolUses: cannot find trait");

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
                                      module);
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

ParseResult MethodCallOp::parse(OpAsmParser& p, OperationState &st) {
  MLIRContext* ctx = p.getContext();

  // grammar:
  //
  // trait.method.call %claim @Trait[Types...]::@method(%arguments...)
  //   :  (Types...) -> Type
  //   as (Types...) -> Type
  //   (by @Proof)?
  //   attr-dict?

  // parse %claim
  OpAsmParser::UnresolvedOperand claim;
  if (p.parseOperand(claim)) return failure();

  // parse '@Trait[Types...]' as TraitApplicationAttr
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp) return failure();

  // parse '::'
  if (p.parseColon() || p.parseColon()) return failure();

  // parse '@method' as FlatSymbolRefAttr
  FlatSymbolRefAttr methodName;
  if (p.parseAttribute(methodName)) return failure();

  // add methodRef attribute
  auto traitName = traitApp.getTraitName().getValue();
  auto methodRef = SymbolRefAttr::get(ctx, traitName, methodName);
  st.addAttribute("method_ref", methodRef);

  // parse '(' %arguments... ')'
  SmallVector<OpAsmParser::UnresolvedOperand> arguments;
  if (p.parseOperandList(arguments, OpAsmParser::Delimiter::Paren)) return failure();

  // parse ':' methodFunctionType
  FunctionType methodFunctionType;
  if (p.parseColonType(methodFunctionType)) return failure();

  // add methodFunctionType attribute
  st.addAttribute("method_function_type", TypeAttr::get(methodFunctionType));

  // parse 'as'
  if (p.parseKeyword("as")) return failure();

  // parse operand types and result type as a FunctionType
  FunctionType argumentTypesAndResultType;
  if (p.parseType(argumentTypesAndResultType)) return failure();

  // add the result types
  st.addTypes(argumentTypesAndResultType.getResults());

  // parse optional 'by' @ProofSym
  FlatSymbolRefAttr proofSym;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(proofSym)) return failure();
  }

  // build the type of %claim
  auto loc = p.getCurrentLocation();
  auto errFn = [&] { return p.emitError(loc); };
  ClaimType claimTy = ClaimType::getChecked(errFn, ctx, traitApp, proofSym);
  if (!claimTy) return failure();

  // resolve %claim
  if (p.resolveOperand(claim, claimTy, st.operands))
    return failure();

  // resolve arguments
  auto argumentTypes = argumentTypesAndResultType.getInputs();
  if (argumentTypes.size() != arguments.size())
    return p.emitError(loc, "argument count mismatch");

  if (p.resolveOperands(arguments, argumentTypes, loc, st.operands))
    return failure();

  // parse attributes
  if (p.parseOptionalAttrDictWithKeyword(st.attributes)) return failure();
  
  return success();
}

void MethodCallOp::print(OpAsmPrinter& p) {
  // grammar:
  //
  // trait.method.call %claim @Trait[Types...]::@method(%arguments...)
  //   :  (Types...) -> Type
  //   as (Types...) -> Type
  //   (by @Proof)?
  //   attr-dict?

  // print %claim
  p << " " << getClaim() << " ";

  // print '@Trait[Types...]'
  getTraitApplication().print(p);

  // '::@method(%arguments...)'
  p << "::" << getMethodAttr() << "(" << getArguments() << ")";

  // on a newline:
  // ': ' methodFunctionType
  p.printNewline();
  p.getStream().indent(2);
  p << ": " << getMethodFunctionType();

  // on a newline:
  // 'as' (argumentTypes) -> (resultTypes)`
  p.printNewline();
  p.getStream().indent(2);
  FunctionType actualFunctionType = FunctionType::get(
    getContext(),
    ValueRange(getArguments()).getTypes(),
    getResultTypes()
  );
  p << "as " << actualFunctionType;

  // on a newline:
  // (by @Proof)?
  if (getClaimType().isProven()) {
    p.printNewline();
    p.getStream().indent(2);
    p << "by " << getClaimType().getProof();
  }

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"method_ref", "method_function_type"}
  );
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

  FunctionType formal = getCalleeFunctionType();
  FunctionType actual = FunctionType::get(getContext(), getOperandTypes(), getResultTypes());

  ModuleOp moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (failed(substituteWith(formal, actual, moduleOp, result)))
    llvm_unreachable("FuncCallOp::buildSubstitution: substituteWith failed");

  return normalizeSubstitution(result);
}

std::string FuncCallOp::getNameOfCalleeInstance() {
  return getCallee().str() +
         generateMangledNameSuffixFor(buildSubstitution(), getCalleeTypeParams());
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
  // parse `%src : @SrcTrait[Types...] (by @SrcProof)? to @DstTrait[Types...] (by @DstProof)?`

  // %src
  OpAsmParser::UnresolvedOperand src;
  if (p.parseOperand(src)) return failure();
  if (p.parseColon()) return failure();

  // @SrcTrait[...]
  TraitApplicationAttr srcApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!srcApp) return failure();

  // (by @SrcProof)?
  FlatSymbolRefAttr srcProof;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(srcProof))
      return failure();
  }

  // resolve %src with the appropriate claim type
  ClaimType srcTy = srcProof
    ? ClaimType::get(p.getContext(), srcApp, srcProof)
    : ClaimType::get(p.getContext(), srcApp);

  if (p.resolveOperand(src, srcTy, st.operands))
    return failure();

  // to
  if (p.parseKeyword("to"))
    return failure();

  // @DstTrait[...]
  TraitApplicationAttr dstApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!dstApp) return failure();

  // (by @DstProof)?
  FlatSymbolRefAttr dstProof;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(dstProof))
      return failure();
  }

  // result type is the claim of the result application
  ClaimType dstTy = dstProof
    ? ClaimType::get(p.getContext(), dstApp, dstProof)
    : ClaimType::get(p.getContext(), dstApp);
  st.addTypes(dstTy);

  return success();
}

void ProjectOp::print(OpAsmPrinter& p) {
  // print `%src: %Trait1[Types...] to @Trait2[Types...]1

  p << " ";

  // Source: %src: @SrcTrait[...] (by @SrcProof)?
  p.printOperand(getSource());
  p << ": ";
  ClaimType srcTy = getSourceClaim();
  srcTy.getTraitApplication().print(p);

  if (srcTy.isProven())
    p << " by " << srcTy.getProof();

  // Destination: to @DstTrait[...] (by @DstProof)?
  p << " to ";
  ClaimType dstTy = getResultClaim();
  dstTy.getTraitApplication().print(p);

  if (dstTy.isProven())
    p << " by " << dstTy.getProof();
}

TraitOp ProjectOp::getSourceTrait() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("ProjectOp::getSourceTrait: not in a module");
  return getSourceClaim().getTraitApplication().getTraitOrAbort(module, "ProjectOp::getSourceTrait: couldn't find trait");
}

TraitOp ProjectOp::getProjectedTrait() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("ProjectOp::getProjectedTrait: not in a module");
  return getResultClaim().getTraitApplication().getTraitOrAbort(module, "ProjectOp::getProjectedTrait: couldn't find trait");
}

LogicalResult ProjectOp::verifySymbolUses(SymbolTableCollection &/*symbolTable*/) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  auto errFn = [&]{ return emitOpError(); };

  // verify source claim
  ClaimType src = getSourceClaim();
  if (failed(src.verifySymbolUses(module, errFn)))
    return failure();

  // verify destination claim
  ClaimType dst = getResultClaim();
  if (failed(dst.verifySymbolUses(module, errFn)))
    return failure();

  // verify proofness parity
  bool srcProven = src.isProven();
  bool dstProven = dst.isProven();
  if (srcProven != dstProven) {
    if (!srcProven)
      return emitOpError() << "result cannot have 'by' when source has no 'by'";
    return emitOpError() << "result must have 'by' when source has 'by'";
  }

  // get candidate projections
  SmallVector<ClaimType> candidates;
  src.getProjections(module, candidates);

  // any matching candidate will do
  for (auto cand : candidates) {
    if (succeeded(substituteWith(cand, dst, module)))
      return success();
  }

  // no matching candidate found
  return emitOpError()
         << "projected claim " << dst
         << "does not unify with any candidate projection of " << src;
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
  getClaim().getTraitApplication().print(p);
}

LogicalResult AllegeOp::verify() {
  // claim must be monomorphic
  if (!getClaim().isMonomorphic())
    return failure();

  return success();
}

LogicalResult AllegeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";
  return getClaim().verifySymbolUses(module, [this] { return emitOpError(); });
}
