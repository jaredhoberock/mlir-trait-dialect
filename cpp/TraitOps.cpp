// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Instantiation.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/Support/xxhash.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <optional>
#include <variant>

#include "TraitOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "TraitOps.cpp.inc"

using namespace mlir;
using namespace mlir::trait;


/// Hash a string into a short '_h{16 hex digits}' suffix.
static std::string hashToSuffix(StringRef input) {
  uint64_t hash = llvm::xxHash64(input);
  std::string result;
  llvm::raw_string_ostream out(result);
  out << llvm::format("_h%016" PRIx64, hash);
  out.flush();
  return result;
}

// Generates a mangled name suffix
// based on the substitution of some polymorphic entity (e.g., ImplOp, FuncOp, etc.)
static std::string generateMangledNameSuffixFor(
  const DenseMap<Type,Type> &subst,
  ArrayRef<GenericTypeInterface> typeParams) {

  // Monomorphic entities have no type parameters to disambiguate,
  // so no suffix is needed. The base name is already unique
  if (typeParams.empty()) return "";

  // Build a full type string for hashing (preserves uniqueness),
  // but only emit a short hash in the actual suffix.
  std::string full;
  llvm::raw_string_ostream os(full);
  for (auto ty : typeParams) {
    os << "_" << applySubstitutionToFixedPoint(subst, ty);
  }
  os.flush();

  return hashToSuffix(full);
}


//===----------------------------------------------------------------------===//
// TraitOp
//===----------------------------------------------------------------------===//

LogicalResult TraitOp::verify() {
  auto typeParams = getTypeParams().getAsValueRange<TypeAttr>();

  // types must be unique GenericTypeParameters
  DenseSet<Type> uniqueParams;
  for (Type ty : typeParams) {
    if (!isa<GenericTypeInterface>(ty))
      return emitOpError() << "expected GenericTypeInterface (e.g., !trait.poly), found " << ty;
    if (!uniqueParams.insert(ty).second)
      return emitOpError() << "type parameters must be unique";
  }

  // there must be at least one type parameter
  if (uniqueParams.size() < 1)
    return emitOpError() << "requires at least one type parameter";

  // collect GAT poly vars from AssocTypeOp type_params
  DenseSet<Type> gatParams;
  for (Operation &op : getBody().front()) {
    if (auto assoc = dyn_cast<AssocTypeOp>(op)) {
      if (auto tp = assoc.getTypeParams()) {
        for (Attribute tyAttr : *tp)
          gatParams.insert(cast<TypeAttr>(tyAttr).getValue());
      }
    }
  }

  // check requirements
  for (auto& app : getRequirements()) {
    // each TraitApplicationAttr must use at least one of the trait's type parameters
    // OR at least one GAT type parameter
    bool mentionsTraitParam = llvm::any_of(uniqueParams, [&](Type param) {
      return app.mentionsType(param);
    });
    bool mentionsGatParam = llvm::any_of(gatParams, [&](Type param) {
      return app.mentionsType(param);
    });

    if (!mentionsTraitParam && !mentionsGatParam)
      return emitOpError() << "'where' clause requirement " << app
                           << " must mention at least one type parameter";

    // A direct self-reference like @Trait[!S] would create a circular
    // obligation that no impl can satisfy. However, a self-reference whose
    // self argument goes through a projection (e.g. @Trait[!trait.proj<...>])
    // is safe: the projection resolves to a concrete type during
    // monomorphization, so the obligation is discharged against a different
    // impl, not the one being defined.
    if (app.getTraitName().getValue() == getSymName()) {
      bool selfArgHasProjection = containsType<ProjectionType>(app.getTypeArgs().front());
      if (!selfArgHasProjection)
        return emitOpError() << "'where' clause requirement " << app
                             << " must not reference the current trait";
    }
  }

  return success();
}

LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // verify obligations
  return getRequirements().verifyTraitApplications(getParentOp<ModuleOp>(), [&](){ return emitOpError(); });
}

FailureOr<DenseMap<Type,Type>> TraitOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                      llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();
  return buildSpecializationSubstitution(getSelfClaim(), actualSelfClaim, *module, errFn);
}


SmallVector<ClaimType> TraitOp::getRequirementsAsClaims() {
  MLIRContext *ctx = getContext();
  return llvm::map_to_vector(getRequirements(), [ctx](TraitApplicationAttr app) {
    return ClaimType::get(ctx, app);
  });
}

FailureOr<SmallVector<ClaimType>> TraitOp::specializeRequirementsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // build a specialized substitution for actualSelfClaim
  auto subst = buildSpecializationSubstitution(getSelfClaim(), actualSelfClaim, *module, errFn);
  if (failed(subst)) return failure();

  // apply the substitution to each requirement
  return llvm::map_to_vector(getRequirementsAsClaims(), [&](ClaimType req) {
    ClaimType specializedReq = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(*subst, req));
    if (!specializedReq)
      llvm_unreachable("TraitOp::specializeRequirementsAsClaimsFor: expected ClaimType");
    return specializedReq;
  });
}

SmallVector<ImplOp> TraitOp::getImpls() {
  auto module = getModule();
  if (failed(module)) return {};

  // traverse users of this trait
  auto uses = mlir::SymbolTable::getSymbolUses(*this, *module);
  if (!uses) return {};

  SmallVector<ImplOp> result;
  for (const auto& use : *uses) {
    if (auto impl = dyn_cast<ImplOp>(use.getUser())) {
      if (impl.getTrait() == *this)
        result.push_back(impl);
    }
  }

  return result;
}

SmallVector<ImplOp> TraitOp::getCandidateImplsFor(ClaimType wanted) {
  SmallVector<ImplOp> result;
  for (auto impl : getImpls()) {
    if (succeeded(impl.buildSubstitutionForSelfClaim(wanted)))
      result.push_back(impl);
  }
  return result;
}

ParseResult TraitOp::parse(OpAsmParser &p, OperationState &s) {
  MLIRContext *ctx = p.getContext();

  // sym_name 
  StringAttr symName;
  if (p.parseSymbolName(symName, "sym_name", s.attributes))
    return failure();

  // [ type_params ]
  SmallVector<Type> typeParams;
  if (failed(p.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, [&] {
        Type ty;
        if (p.parseType(ty)) return failure();
        typeParams.push_back(ty);
        return success();
      })))
    return failure();

  // build TypeArrayAttr
  SmallVector<Attribute,4> typeAttrs;
  typeAttrs.reserve(typeParams.size());
  for (auto ty : typeParams) {
    typeAttrs.push_back(TypeAttr::get(ty));
  }
  s.addAttribute("type_params", ArrayAttr::get(ctx, typeAttrs));

  // requirements
  auto requirements = TraitApplicationArrayAttr::get(ctx, {});
  if (succeeded(p.parseOptionalKeyword("where"))) {
    requirements = dyn_cast_or_null<TraitApplicationArrayAttr>(TraitApplicationArrayAttr::parse(p,{}));
    if (!requirements)
      return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationArrayAttr");
  }
  s.addAttribute("requirements", requirements);

  // attr-dict-with-keyword
  if (p.parseOptionalAttrDictWithKeyword(s.attributes))
    return failure();

  // region body
  Region *body = s.addRegion();
  if (p.parseRegion(*body, /*args=*/{}, /*types=*/{})) return failure();
  if (body->empty()) body->emplaceBlock();

  return success();
}

void TraitOp::print(OpAsmPrinter &p) {
  // `@sym_name`
  p << ' ';
  p.printSymbolName(getSymNameAttr());

  // `[ type_params ]`
  p << "[";
  llvm::interleaveComma(getTypeParams(), p, [&](Attribute tyAttr) {
    p.printType(cast<TypeAttr>(tyAttr).getValue());
  });
  p << ']';

  // print requirements if not empty
  if (hasRequirements()) {
    p << " where ";
    getRequirements().print(p);
  }

  // print any trailing attributes
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elided=*/{"sym_name","type_params","requirements"});

  // region body
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}


//===----------------------------------------------------------------------===//
// ImplOp
//===----------------------------------------------------------------------===//

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&]{ return emitOpError(); };

  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // Verify self application attribute exists
  auto selfApp = getSelfApplication();
  if (!selfApp)
    return emitOpError() << "requires a self application TraitApplicationAttr";

  // Verify the self application
  if (failed(selfApp.verifyTraitApplication(*module, errFn)))
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

  // Verify methods and associated type bindings
  llvm::SmallSet<StringRef, 8> definedMethods;
  llvm::SmallSet<StringRef, 8> definedAssocTypes;
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

      // Verify that the impl method's signature matches the trait method's signature
      auto traitMethod = traitOp.getMethod(name, errFn);
      if (failed(traitMethod)) return failure();

      // Build substitution from trait type params to impl type args
      auto traitSubst = traitOp.buildSubstitutionForSelfClaim(getSelfClaim(), errFn);
      if (failed(traitSubst)) return failure();

      // Specialize the trait method's signature
      FunctionType traitMethodTy = traitMethod->getFunctionType();
      Type specializedTraitMethodTy = applySubstitutionToFixedPoint(*traitSubst, traitMethodTy);

      // Resolve any ProjectionTypes in the specialized signature using this impl's bindings
      specializedTraitMethodTy = resolveProjectionTypesViaBindings(specializedTraitMethodTy, *traitSubst);

      // Check that the impl method's signature can specialize to the expected signature
      FunctionType implMethodTy = implMethod.getFunctionType();
      if (failed(buildSpecializationSubstitution(specializedTraitMethodTy, implMethodTy, *module, errFn))) {
        return emitOpError() << "method '" << name << "' has incompatible signature: "
                             << "expected " << specializedTraitMethodTy
                             << " but found " << implMethodTy;
      }
    } else if (auto assocType = dyn_cast<AssocTypeOp>(op)) {
      StringRef name = assocType.getSymName();
      if (!definedAssocTypes.insert(name).second)
        return emitOpError() << "defines associated type '" << name << "' multiple times";

      // In an impl, the associated type must have a bound_type
      if (!assocType.getBoundType())
        return emitOpError() << "associated type '" << name << "' in impl must have a bound type";

      // Verify that the trait declares this associated type
      auto traitAssoc = traitOp.getAssociatedType(name);
      if (failed(traitAssoc))
        return emitOpError() << "associated type '" << name
                             << "' not found in trait '" << getTraitNameAttr() << "'";

      // Verify GAT type_params arity matches
      {
        unsigned traitArity = traitAssoc->getTypeParams() ? traitAssoc->getTypeParams()->size() : 0;
        unsigned implArity = assocType.getTypeParams() ? assocType.getTypeParams()->size() : 0;
        if (traitArity != implArity)
          return emitOpError() << "associated type '" << name
                               << "' has " << implArity << " type parameter(s) but trait declares "
                               << traitArity;
      }
    } else {
      return emitOpError() << "body may only contain 'func.func' or 'trait.assoc_type' operations";
    }
  }

  // Verify that all associated types in the trait have bindings in the impl
  for (auto traitAssoc : traitOp.getAssociatedTypes()) {
    if (!definedAssocTypes.contains(traitAssoc.getSymName()))
      return emitOpError() << "missing binding for associated type '"
                           << traitAssoc.getSymName()
                           << "' of trait '" << getTraitNameAttr() << "'";
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

bool ImplOp::isUnconditional() {
  // an ImplOp is unconditional if:
  // 1. it is monomorphic (no type parameters),
  // 2. its TraitOp has no requirements, and
  // 3. it has no assumptions
  return getTypeParams().empty() &&
         getAssumptions().empty() &&
         getTrait().getRequirements().empty();
}

LogicalResult ImplOp::verifyIsUnconditional(llvm::function_ref<InFlightDiagnostic()> err) {
  if (!isUnconditional()) {
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

FailureOr<DenseMap<Type,Type>> ImplOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                     llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();
  return buildSpecializationSubstitution(getSelfClaim(), actualSelfClaim, *module, errFn);
}

FailureOr<Type> ImplOp::specializeAssociatedTypeBinding(
    StringRef name,
    ArrayRef<Type> assocTypeArgs,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto binding = getAssociatedTypeBinding(name, err);
  if (failed(binding)) return failure();

  auto assoc = getAssociatedType(name);
  if (succeeded(assoc) && assoc->getTypeParams()) {
    auto typeParams = *assoc->getTypeParams();
    if (typeParams.size() != assocTypeArgs.size()) {
      if (err) err() << "GAT arity mismatch for '" << name
                     << "': expected " << typeParams.size()
                     << " type args but got " << assocTypeArgs.size();
      return failure();
    }
    *binding = applyGATSubstitution(typeParams, assocTypeArgs, *binding);
  }

  return *binding;
}

Type ImplOp::resolveProjectionTypesViaBindings(Type ty, const DenseMap<Type,Type> &subst) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](Type t) -> std::optional<Type> {
    auto proj = dyn_cast<ProjectionType>(t);
    if (!proj) return std::nullopt;
    auto resolved = specializeAssociatedTypeBinding(
        proj.getAssocName().getValue(), proj.getAssocTypeArgs());
    if (failed(resolved)) return std::nullopt;
    return applySubstitutionToFixedPoint(subst, *resolved);
  });
  return replacer.replace(ty);
}

FailureOr<DenseMap<Type,Type>> ImplOp::buildMonomorphizationSubstitutionFor(
    ClaimType provenSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (!provenSelfClaim.isProven()) {
    if (err) err() << "expected proven self claim for " << getSymName();
    return failure();
  }

  auto module = getModule(err);
  if (failed(module)) return failure();

  DenseMap<Type,Type> subst;

  // bind the *same* self claim without a proof to the proven self claim
  // this (recursively) records claim -> proven-claim into subst
  ClaimType unprovenSelfClaim = provenSelfClaim.asUnproven();
  if (failed(verifyAndRecordProof(unprovenSelfClaim, provenSelfClaim, *module, subst, err)))
    return failure();

  // add PolyType -> concrete Type bindings for monomorphization
  auto polyToType = buildSubstitutionForSelfClaim(provenSelfClaim, err);
  if (failed(polyToType)) return failure();

  // merge polyToType into subst; flag conflicts
  for (const auto &[k,v] : *polyToType) {
    auto [it, inserted] = subst.try_emplace(k,v);
    if (!inserted && it->second != v) {
      if (err) err() << "conflicting substitution for " << k
                     << ": " << it->second << " vs " << v;
      return failure();
    }
  }

  return normalizeSubstitution(subst);
}

SmallVector<GenericTypeInterface, 4> ImplOp::getTypeParams() {
  // collect all the types where a type variable could hide
  SmallVector<Type> allOurTypes;
  allOurTypes.push_back(getSelfClaim());
  for (ClaimType a : getAssumptionsAsClaims()) {
    allOurTypes.push_back(a);
  }

  // tuple the types
  TupleType tupled = TupleType::get(getContext(), allOurTypes);

  // get all the generic types in the tuple
  return getGenericTypesIn(tupled);
}

FailureOr<func::FuncOp> ImplOp::getOrInstantiateMethod(OpBuilder& builder, StringRef methodName) {
  auto trait = getTrait();

  // check that we've named a valid trait method
  if (!trait.hasMethod(methodName)) return failure();

  // check if the method already exists in the ImplOp
  auto method = getMethod(methodName);
  if (succeeded(method)) return method;

  // otherwise, we need to instantiate the method from the default implementation in the trait
  auto traitMethod = trait.getOptionalMethod(methodName);
  if (failed(traitMethod)) return failure();

  // build a substitution that maps trait PolyType parameters to impl type arguments
  auto subst = trait.buildSubstitutionForSelfClaim(getSelfClaim());
  if (failed(subst)) return failure();

  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&getBody().front());
  return instantiatePolymorph(builder, *traitMethod, methodName, *subst);
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

  // build claim map from function's claim-typed parameters
  DenseMap<TraitApplicationAttr, Value> claimMap;
  for (auto arg : funcOp.getArguments()) {
    if (auto claimTy = dyn_cast<ClaimType>(arg.getType()))
      claimMap[claimTy.getTraitApplication()] = arg;
  }

  // replace all AssumeOps: match against function parameters first,
  // fall back to projection from selfProofArg
  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(a);

    Value replacement;
    auto it = claimMap.find(a.getTraitApplication());
    if (it != claimMap.end()) {
      // matched a function parameter — use it directly
      replacement = it->second;
    } else {
      // fall back to projection from self-proof
      replacement = rewriter.create<ProjectOp>(
        a.getLoc(),
        a.getClaim(),
        selfProofArg
      );
    }

    rewriter.replaceAllUsesWith(a.getResult(), replacement);
    toErase.push_back(a);
  });

  // erase the AssumeOps
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

FailureOr<func::FuncOp> ImplOp::getOrInstantiateFreeFunctionFromMethod(
    PatternRewriter& rewriter,
    ClaimType provenSelfClaim,
    StringRef methodName,
    const DenseMap<Type,Type> &extraSubst) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return failure();

  // generate a free function name based on the ImplOp's mangled name for our proof
  auto functionName = generateMangledName(provenSelfClaim) + "_" + methodName.str();

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
    getParentOp(),
    FlatSymbolRefAttr::get(ctx, functionName)
  );

  if (!funcOp) {
    // get the method inside the ImplOp
    auto method = getOrInstantiateMethod(rewriter, methodName);
    if (failed(method)) return failure();

    // build a poly→concrete substitution from the impl's type parameters
    auto subst = buildMonomorphizationSubstitutionFor(provenSelfClaim);
    if (failed(subst)) return failure();

    // merge caller-provided entries (e.g., projection→concrete bindings)
    for (const auto &[k, v] : extraSubst)
      subst->try_emplace(k, v);

    // instantiate into grandparent with mangled name
    funcOp = instantiateMethodAsFreeFuncWithLeadingSelfProof(
      rewriter,
      getParentOp(),
      *method,
      functionName,
      provenSelfClaim,
      *subst
    );
  }

  return funcOp;
}

/// Generate a deterministic symbol name for an ImplOp.
/// 
/// The name has the form {TraitName}_impl_h{hash} where the hash is a
/// 64-bit xxHash of the full type argument and assumption signature. This
/// keeps symbols short and bounded in length.
std::string ImplOp::generateSymName(TraitApplicationAttr selfApp,
                                    TraitApplicationArrayAttr assumptions) {
  // Build the full type argument and assumption signature for hashing
  std::string signature;
  llvm::raw_string_ostream os(signature);
  for (auto ty : selfApp.getTypeArgs()) {
    os << "_" << ty;
  }
  if (!assumptions.empty()) {
    os << "_where";
    for (auto app : assumptions) {
      os << "_" << app.getTraitName().getValue();
      for (auto typeArg : app.getTypeArgs()) {
        os << "_" << typeArg;
      }
    }
  }
  os.flush();

  return selfApp.getTraitName().getValue().str() + "_impl" + hashToSuffix(signature);
}

std::string ImplOp::generateMangledName(ClaimType claim) {
  auto subst = buildSubstitutionForSelfClaim(claim);
  if (failed(subst))
    llvm_unreachable("ImplOp::generateMangledName: specializedSelfClaimAgainst failed");

  std::string result = getSymName().str();
  result += generateMangledNameSuffixFor(*subst, getTypeParams());
  return result;
}

SmallVector<ClaimType> ImplOp::getAssumptionsAsClaims() {
  MLIRContext *ctx = getContext();
  return llvm::map_to_vector(getAssumptions(), [ctx](TraitApplicationAttr app) {
    return ClaimType::get(ctx, app);
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeAssumptionsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // build a specialized substitution for actualSelfClaim
  auto subst = buildSpecializationSubstitution(getSelfClaim(), actualSelfClaim, *module, errFn);
  if (failed(subst)) return failure();

  // apply the substitution to each assumption
  return llvm::map_to_vector(getAssumptionsAsClaims(), [&](ClaimType assumption) {
    ClaimType specializedAssumption = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(*subst, assumption));
    if (!specializedAssumption)
      llvm_unreachable("ImplOp::specializeAssumptionsAsClaimsFor: expected ClaimType");
    return specializedAssumption;
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeObligationsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // specialize requirements of the trait
  auto requirements = getTrait().specializeRequirementsAsClaimsFor(actualSelfClaim, errFn);
  if (failed(requirements)) return failure();

  // resolve projections in requirements using this impl's associated type
  // bindings (e.g., `Coord[Tensor[Self]::Shape]` becomes `Coord[tuple<i64,i64>]`
  // when the impl binds `Shape = S` and S is specialized to tuple<i64,i64>)
  auto subst = buildSubstitutionForSelfClaim(actualSelfClaim, errFn);
  if (failed(subst)) return failure();

  for (ClaimType &req : *requirements)
    req = cast<ClaimType>(resolveProjectionTypesViaBindings(req, *subst));

  // specialize assumptions of the impl
  auto assumptions = specializeAssumptionsAsClaimsFor(actualSelfClaim, errFn);
  if (failed(assumptions)) return failure();

  // obligations = requirements + assumptions
  SmallVector<ClaimType> obligations = std::move(*requirements);
  obligations.append(std::move(*assumptions));

  return obligations;
}

ParseResult ImplOp::parse(OpAsmParser &p, OperationState &result) {
  // parse optional symbolic name: trait.impl @Sym
  StringAttr parsedSymName;
  (void)p.parseOptionalSymbolName(parsedSymName);

  // parse mandatory for
  if (p.parseKeyword("for"))
    return failure();
  
  // parse @TraitName[Types...]
  TraitApplicationAttr selfApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!selfApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addAttribute("self_application", selfApp);  
  
  // assumptions
  auto assumptions = TraitApplicationArrayAttr::get(p.getContext(), {});
  if (succeeded(p.parseOptionalKeyword("where"))) {
    assumptions = dyn_cast_or_null<TraitApplicationArrayAttr>(TraitApplicationArrayAttr::parse(p, {}));
    if (!assumptions)
      return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationArrayAttr");
  }
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

  // print assumptions if not empty
  if (!getAssumptions().empty()) {
    printer << "where ";
    getAssumptions().print(printer);
  }
  
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

  // verify basic symbol uses of the claim
  if (failed(getProvenClaim().verifySymbolUses(module, errFn)))
    return failure();

  // check that the named impl exists
  auto implOp = getImpl();
  if (!implOp)
    return emitOpError() << "cannot find impl '" << getImplNameAttr() << "'";

  // check that we are able to substitute the impl's self claim against our proven claim
  if (failed(implOp.buildSubstitutionForSelfClaim(getProvenClaim(), errFn)))
    return failure();

  // recursively verify proof structure and that proof bindinds can be recorded
  DenseMap<Type,Type> subst;
  if (failed(verifyAndRecordProof(getProvenClaim().asUnproven(), getProvenClaim(), module, subst, errFn)))
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

    // reject self-referencing sub-proofs
    if (subproofRef.getValue() == getSymName()) {
      if (err) err() << "sub-proof '" << subproofRef
                     << "' must not reference the proof itself";
      return failure();
    }

    auto subproof = getProofOpOrUnconditionalImplOp(module, subproofRef, err);
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

/// Look up a proof symbol and return the raw Operation* (ProofOp or ImplOp).
/// This is the shared lookup used by both getImplFromProof and
/// getProofOpOrUnconditionalImplOp.
static FailureOr<Operation*> lookupProofSymbol(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  Operation* symOp = SymbolTable::lookupNearestSymbolFrom(module, name);
  if (!symOp) {
    if (errFn) errFn() << "cannot find proof symbol '" << name << "'";
    return failure();
  }

  if (isa<ImplOp>(symOp) || isa<ProofOp>(symOp))
    return symOp;

  if (errFn) errFn() << "proof symbol '" << name << "' must refer to trait.proof or trait.impl";
  return failure();
}

FailureOr<ImplOp> ProofOp::getImplFromProof(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto symOp = lookupProofSymbol(module, name, errFn);
  if (failed(symOp)) return failure();

  if (auto implOp = dyn_cast<ImplOp>(*symOp))
    return implOp;

  auto proofOp = cast<ProofOp>(*symOp);
  ImplOp impl = proofOp.getImpl();
  if (!impl) {
    if (errFn) errFn() << "proof '" << name << "' does not resolve to an impl";
    return failure();
  }
  return impl;
}

FailureOr<Operation*> ProofOp::getProofOpOrUnconditionalImplOp(
    ModuleOp module,
    FlatSymbolRefAttr name,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto symOp = lookupProofSymbol(module, name, errFn);
  if (failed(symOp)) return failure();

  // if it's an ImplOp, it must be unconditional
  if (auto impl = dyn_cast<ImplOp>(*symOp)) {
    if (failed(impl.verifyIsUnconditional(errFn))) return failure();
  }

  return *symOp;
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

  // verify the claim type
  if (failed(getProvenClaim().verifySymbolUses(module, errFn)))
    return failure();

  // look up the proof symbol (must be ProofOp or unconditional ImplOp)
  auto symOp = ProofOp::getProofOpOrUnconditionalImplOp(module, getProofAttr(), errFn);
  if (failed(symOp)) return failure();

  // check that the underlying impl can build a substitution for our claim
  auto impl = ProofOp::getImplFromProof(module, getProofAttr(), errFn);
  if (failed(impl)) return failure();

  return impl->buildSubstitutionForSelfClaim(getProvenClaim(), errFn);
}


//===----------------------------------------------------------------------===//
// DeriveOp
//===----------------------------------------------------------------------===//

ParseResult DeriveOp::parse(OpAsmParser &p, OperationState &result) {
  // trait.derive @Trait[Types...] from @Impl given(%claims...)

  // parse @Trait[Types...]
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp)
    return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");
  result.addAttribute("trait_application", traitApp);

  // parse `from`
  if (p.parseKeyword("from"))
    return failure();

  // parse @Impl
  FlatSymbolRefAttr implRef;
  if (p.parseAttribute(implRef, "impl", result.attributes))
    return failure();

  // parse `given`
  if (p.parseKeyword("given"))
    return failure();

  // parse ( %claims... )
  SmallVector<OpAsmParser::UnresolvedOperand> assumptions;
  if (p.parseOperandList(assumptions, OpAsmParser::Delimiter::Paren))
    return failure();

  // parse `: (` type_list `)`
  SmallVector<Type> assumptionTypes;
  if (!assumptions.empty()) {
    if (p.parseColon())
      return failure();
    if (failed(p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&] {
          Type ty;
          if (p.parseType(ty)) return failure();
          assumptionTypes.push_back(ty);
          return success();
        })))
      return failure();

    if (assumptionTypes.size() != assumptions.size())
      return p.emitError(p.getCurrentLocation(), "assumption type count mismatch");

    auto loc = p.getCurrentLocation();
    if (p.resolveOperands(assumptions, assumptionTypes, loc, result.operands))
      return failure();
  }

  // construct the unproven result type
  ClaimType claimTy = ClaimType::get(p.getContext(), traitApp);
  result.addTypes(claimTy);

  // parse optional attributes
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void DeriveOp::print(OpAsmPrinter &p) {
  // trait.derive @Trait[Types...] from @Impl given(%claims...)

  p << " ";
  getTraitApplication().print(p);
  p << " from " << getImplAttr() << " given(";
  llvm::interleaveComma(getAssumptions(), p, [&](Value v) {
    p.printOperand(v);
  });
  p << ")";

  // print types if there are assumptions
  if (!getAssumptions().empty()) {
    p << " : (";
    llvm::interleaveComma(getAssumptions().getTypes(), p, [&](Type ty) {
      p.printType(ty);
    });
    p << ")";
  }

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"trait_application", "impl"}
  );
}

ImplOp DeriveOp::getImplOp() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;
  return mlir::SymbolTable::lookupNearestSymbolFrom<ImplOp>(module, getImplAttr());
}

/// Verifies that a trait.derive op is well-formed with respect to its symbols:
///
///  1. The @impl symbol resolves to a trait.impl op.
///  2. The impl's self application can be specialized against the derived claim
///     (i.e., the impl's header structurally matches the claim we want to derive).
///  3. The number of assumption operands equals the impl's assumption count
///     after specialization.
///  4. Each operand's claim type matches the corresponding specialized
///     assumption (so the caller is providing exactly the evidence the impl
///     requires under this specialization).
LogicalResult DeriveOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&] { return emitOpError(); };

  // look up impl by symbol
  auto implOp = getImplOp();
  if (!implOp)
    return emitOpError() << "cannot find trait.impl '" << getImplAttr() << "'";

  // build substitution: impl's self claim -> derived claim
  ClaimType derivedClaim = getDerivedClaim();
  auto subst = implOp.buildSubstitutionForSelfClaim(derivedClaim, errFn);
  if (failed(subst))
    return failure();

  // specialize impl's assumptions for the derived claim
  auto specializedAssumptions = implOp.specializeAssumptionsAsClaimsFor(derivedClaim, errFn);
  if (failed(specializedAssumptions))
    return failure();

  // check operand count matches assumption count
  if (getAssumptions().size() != specializedAssumptions->size())
    return emitOpError() << "expected " << specializedAssumptions->size()
                         << " assumption operands, got " << getAssumptions().size();

  // check each operand's claim type matches the corresponding specialized assumption
  for (auto [i, pair] : llvm::enumerate(llvm::zip(getAssumptions(), *specializedAssumptions))) {
    auto [operand, expected] = pair;
    ClaimType operandClaim = cast<ClaimType>(operand.getType());
    if (operandClaim.getTraitApplication() != expected.getTraitApplication())
      return emitOpError() << "assumption operand #" << i
                           << " has claim " << operandClaim
                           << " but expected " << expected;
  }

  return success();
}


//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

ParseResult AssumeOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `@Trait[Types...]`
  TraitApplicationAttr app = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

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

  // collect all assumable trait applications
  DenseSet<TraitApplicationAttr> assumable;

  // primary: function claim-typed parameters
  for (auto argType : funcOp.getArgumentTypes()) {
    if (auto claimTy = dyn_cast<ClaimType>(argType))
      assumable.insert(claimTy.getTraitApplication());
  }

  // fallback: enclosing trait/impl
  TraitOp enclosingTrait = funcOp->getParentOfType<TraitOp>();
  ImplOp enclosingImpl = funcOp->getParentOfType<ImplOp>();

  if (enclosingTrait)
    assumable.insert(enclosingTrait.getSelfApplication());

  if (enclosingImpl) {
    assumable.insert(enclosingImpl.getSelfApplication());
    for (auto a : enclosingImpl.getAssumptions())
      assumable.insert(a);
  }

  auto assumedApp = getTraitApplication();

  if (!assumable.contains(assumedApp))
    return emitOpError() << "assumed trait application " << assumedApp
                         << " is not assumable in this context";

  return success();
}

TraitOp AssumeOp::getTrait() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("AssumeOp:getTrait: not inside of a module");
  return getTraitApplication().getTraitOrAbort(module, "AssumeOp::getTrait: couldn't find trait");
}


//===----------------------------------------------------------------------===//
// ProjCastOp
//===----------------------------------------------------------------------===//

// Format: trait.proj.cast %input, %claim : input_type to result_type
ParseResult ProjCastOp::parse(OpAsmParser &p, OperationState &st) {
  OpAsmParser::UnresolvedOperand input, claim;

  // parse '%input, %claim'
  if (p.parseOperand(input) || p.parseComma() || p.parseOperand(claim))
    return failure();

  // parse ': input_type'
  Type inputType;
  if (p.parseColon() || p.parseType(inputType))
    return failure();

  // parse 'to result_type'
  Type resultType;
  if (p.parseKeyword("to") || p.parseType(resultType))
    return failure();

  // parse 'by !trait.claim<...>'
  Type claimType;
  if (p.parseKeyword("by") || p.parseType(claimType))
    return failure();

  st.addTypes(resultType);

  // resolve input
  if (p.resolveOperand(input, inputType, st.operands))
    return failure();

  // resolve claim
  if (p.resolveOperand(claim, claimType, st.operands))
    return failure();

  return success();
}

void ProjCastOp::print(OpAsmPrinter &p) {
  // trait.proj.cast %input, %claim : input_type to result_type by !trait.claim<...>
  p << " " << getInput() << ", " << getClaim()
    << " : " << getInput().getType()
    << " to " << getResult().getType()
    << " by " << getClaim().getType();
}

LogicalResult ProjCastOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitError() << "not inside a module";

  auto errFn = [&] { return emitOpError(); };

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());

  // Structural check: at least one type must contain a ProjectionType
  if (!containsType<ProjectionType>(inputType) &&
      !containsType<ProjectionType>(resultType))
    return emitOpError() << "at least one of input/result must contain "
                         << "a !trait.proj type";

  // If the claim is unproven, defer further checking to monomorphization
  if (!claimTy.isProven())
    return success();

  // Proven claim: resolve matching projections and verify equivalence
  auto implOr = ProofOp::getImplFromProof(module, claimTy.getProof(), errFn);
  if (failed(implOr)) return failure();

  auto subst = implOr->buildSubstitutionForSelfClaim(claimTy.asUnproven(), errFn);
  if (failed(subst)) return failure();

  // Resolve matching projections on both sides
  Type resolvedInput = implOr->resolveProjectionTypesViaBindings(inputType, *subst);
  Type resolvedResult = implOr->resolveProjectionTypesViaBindings(resultType, *subst);

  // If either side still contains unresolved projections (from a different trait),
  // we can't fully verify — defer to monomorphization
  if (containsType<ProjectionType>(resolvedInput) ||
      containsType<ProjectionType>(resolvedResult))
    return success();

  if (resolvedInput != resolvedResult)
    return emitOpError() << "resolved input type " << resolvedInput
                         << " does not match resolved result type " << resolvedResult;

  return success();
}


//===----------------------------------------------------------------------===//
// MethodCallOp
//===----------------------------------------------------------------------===//

FailureOr<TraitOp> MethodCallOp::getTrait(llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();
  return getClaimType()
    .getTraitApplication()
    .getTrait(*module, err);
}

FailureOr<func::FuncOp> MethodCallOp::getMethod(llvm::function_ref<InFlightDiagnostic()> err) {
  auto maybeTrait = getTrait(err);
  if (failed(maybeTrait)) return failure();
  auto func = maybeTrait->getMethod(getMethodName(), err);
  if (failed(func)) {
    return failure();
  }
  return func;
}

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
  auto errFn = [&]{ return emitOpError(); };

  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // verify basics about the claim
  ClaimType claim = getClaimType();
  if (failed(claim.verifySymbolUses(*module, errFn)))
    return failure();

  // check that we can build a consistent substitution for this method call
  return buildParameterSpecialization(errFn);
}

FailureOr<DenseMap<Type,Type>> MethodCallOp::buildParameterSpecialization(llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  auto trait = getTrait(err);
  if (failed(trait)) return failure();

  auto methodFormalTy = getMethodFunctionType(err);
  if (failed(methodFormalTy)) return failure();

  // specialize the method's formal function type by the call's claim
  // this yields the callee's *trait-level* substitution (poly -> type) and
  // applies it to the method signature so that any trait-level generics match our claim
  auto traitSubst = trait->buildSubstitutionForSelfClaim(getClaimType(), err);
  if (failed(traitSubst)) return failure();
  Type formal = applySubstitutionToFixedPoint(*traitSubst, *methodFormalTy);

  // solve the *call-site* specialization: unify the specialized formal type with the
  // actual call type to get any remaining bindings (including generics in args/results)
  Type actual = getActualFunctionType();
  return buildSpecializationSubstitution(formal, actual, *module, err);
}

ImplOp MethodCallOp::getProvenImpl() {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  assert(claimTy.isProven());

  auto module = getModule();
  if (failed(module))
    llvm_unreachable("MethodCallOp::getProvenImpl: not in a module");

  auto impl = ProofOp::getImplFromProof(*module, claimTy.getProof());
  if (failed(impl))
    llvm_unreachable("MethodCallOp::getProvenImpl: getImplFromProof failed");

  return *impl;
}

FailureOr<func::FuncOp> MethodCallOp::getOrInstantiateCallee(
    PatternRewriter &rewriter,
    const DenseMap<Type,Type> &subst) {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  return getProvenImpl()
    .getOrInstantiateFreeFunctionFromMethod(rewriter, claimTy, getMethodName(), subst);
}

ParseResult MethodCallOp::parse(OpAsmParser& p, OperationState &st) {
  MLIRContext* ctx = p.getContext();

  // grammar:
  //
  // trait.method.call %claim @Trait[Types...]::@method(%arguments...)
  //   : (Types...) -> Type
  //   (by @Proof)?
  //   attr-dict?

  // parse %claim
  OpAsmParser::UnresolvedOperand claim;
  if (p.parseOperand(claim)) return failure();

  // parse '@Trait[Types...]' as TraitApplicationAttr
  TraitApplicationAttr traitApp = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!traitApp) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

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
  FunctionType argumentTypesAndResultType;
  if (p.parseColonType(argumentTypesAndResultType)) return failure();

  // add the result types
  st.addTypes(argumentTypesAndResultType.getResults());

  // parse optional 'by' @ProofSym
  FlatSymbolRefAttr proofSym;
  if (succeeded(p.parseOptionalKeyword("by"))) {
    if (p.parseAttribute(proofSym)) return failure();
  }

  // build the type of %claim
  auto loc = p.getCurrentLocation();
  ClaimType claimTy = ClaimType::get(ctx, traitApp, proofSym);

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
  //   : (Types...) -> Type
  //   (by @Proof)?
  //   attr-dict?

  // print %claim
  p << " " << getClaim() << " ";

  // print '@Trait[Types...]'
  getTraitApplication().print(p);

  // '::@method(%arguments...)'
  p << "::" << getMethodAttr() << "(" << getArguments() << ")";

  // on a newline:
  // ': ' (argumentTypes) -> (resultTypes)`
  p.printNewline();
  p.getStream().indent(2);
  FunctionType actualFunctionType = FunctionType::get(
    getContext(),
    ValueRange(getArguments()).getTypes(),
    getResultTypes()
  );
  p << ": " << actualFunctionType;

  // on a newline:
  // (by @Proof)?
  if (getClaimType().isProven()) {
    p.printNewline();
    p.getStream().indent(2);
    p << "by " << getClaimType().getProof();
  }

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"method_ref"}
  );
}


//===----------------------------------------------------------------------===//
// FuncCallOp
//===----------------------------------------------------------------------===//

LogicalResult FuncCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto calleeName = getCalleeNameAttr();
  if (!calleeName)
    return emitOpError() << "requires a 'callee_name' symbol reference attribute";

  auto errFn = [&] { return emitOpError(); };

  auto callee = getCallee(errFn);
  if (failed(callee)) return failure();

  // check that we can build a substitution
  return buildParameterSpecialization(errFn);
}

FailureOr<DenseMap<Type, Type>> FuncCallOp::buildParameterSpecialization(llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  // get formal and actual function types
  auto maybeFormal = getCalleeFunctionType(err);
  if (failed(maybeFormal)) return failure();

  FunctionType formal = *maybeFormal;
  FunctionType actual = getActualFunctionType();

  // build a substitution unifying formal & actual
  auto subst = buildSpecializationSubstitution(formal, actual, *module, err);
  if (failed(subst)) return failure();

  // record any proof bindings found in the actual signature
  if (failed(recordProofBindingsIn(actual, *module, *subst, err)))
    return failure();

  return normalizeSubstitution(*subst);
}

std::string FuncCallOp::getNameOfCalleeInstance() {
  auto subst = buildParameterSpecialization();
  if (failed(subst))
    llvm_unreachable("FuncCallOp::getNameOfCalleeInstance: buildParameterSpecialization failed");

  return getCalleeName().str() +
         generateMangledNameSuffixFor(*subst, getCalleeTypeParams());
}

FailureOr<func::FuncOp> FuncCallOp::getOrInstantiateCallee(
    OpBuilder &builder,
    const DenseMap<Type,Type> &subst) {
  auto module = getModule();
  if (failed(module)) return failure();

  std::string instanceName = getNameOfCalleeInstance();
  auto *symOp = SymbolTable::lookupSymbolIn(*module, builder.getStringAttr(instanceName));
  func::FuncOp existing = dyn_cast_or_null<func::FuncOp>(symOp);
  if (existing) return existing;

  auto callee = getCallee();
  if (failed(callee)) return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(*callee);
  return instantiatePolymorph(builder, *callee, instanceName, subst);
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
  if (!srcApp) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

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
  if (!dstApp) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

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
    if (cand == dst)
      return success();
  }

  // no matching candidate found
  return emitOpError()
         << "projected claim " << dst
         << "is not a candidate projection of " << src;
}


//===----------------------------------------------------------------------===//
// AssocTypeOp
//===----------------------------------------------------------------------===//

ParseResult AssocTypeOp::parse(OpAsmParser &p, OperationState &st) {
  MLIRContext *ctx = p.getContext();

  // parse @Name
  StringAttr symName;
  if (p.parseSymbolName(symName, "sym_name", st.attributes))
    return failure();

  // parse optional <[type_params...]>
  if (succeeded(p.parseOptionalLess())) {
    SmallVector<Type> typeParams;
    if (failed(p.parseCommaSeparatedList(OpAsmParser::Delimiter::Square, [&] {
          Type ty;
          if (p.parseType(ty)) return failure();
          typeParams.push_back(ty);
          return success();
        })))
      return failure();

    if (p.parseGreater())
      return failure();

    SmallVector<Attribute, 4> typeAttrs;
    typeAttrs.reserve(typeParams.size());
    for (Type ty : typeParams)
      typeAttrs.push_back(TypeAttr::get(ty));
    st.addAttribute("type_params", ArrayAttr::get(ctx, typeAttrs));
  }

  // parse optional = bound_type
  if (succeeded(p.parseOptionalEqual())) {
    TypeAttr boundType;
    if (p.parseAttribute(boundType, "bound_type", st.attributes))
      return failure();
  }

  // parse optional attr-dict
  if (p.parseOptionalAttrDict(st.attributes))
    return failure();

  return success();
}

void AssocTypeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymNameAttr());

  // print <[type_params...]> if present
  if (auto tp = getTypeParams(); tp && !tp->empty()) {
    p << "<[";
    llvm::interleaveComma(*tp, p, [&](Attribute tyAttr) {
      p.printType(cast<TypeAttr>(tyAttr).getValue());
    });
    p << "]>";
  }

  // print = bound_type if present
  if (auto bt = getBoundType()) {
    p << " = " << *bt;
  }

  // print any trailing attributes
  p.printOptionalAttrDict((*this)->getAttrs(),
                           /*elided=*/{"sym_name", "bound_type", "type_params"});
}


//===----------------------------------------------------------------------===//
// AllegeOp
//===----------------------------------------------------------------------===//

ParseResult AllegeOp::parse(OpAsmParser &p, OperationState &st) {
  // parse `@Trait[Types...]`
  TraitApplicationAttr app = dyn_cast_or_null<TraitApplicationAttr>(TraitApplicationAttr::parse(p, {}));
  if (!app) return p.emitError(p.getCurrentLocation(), "expected a TraitApplicationAttr");

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
    return emitOpError() << "expected monomorphic claim, got "
                         << getClaim();
  return success();
}

LogicalResult AllegeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";
  return getClaim().verifySymbolUses(module, [this] { return emitOpError(); });
}
