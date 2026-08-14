// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Specialization.hpp"
#include "Trait.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/STLForwardCompat.h>
#include <llvm/Support/xxhash.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/RegionKindInterface.h>
#include <optional>
#include <variant>


#define GET_OP_CLASSES
#include <TraitOps.cpp.inc>

using namespace mlir;
using namespace mlir::trait;

namespace mlir::trait { std::string hashToSuffix(StringRef input); }

namespace {

/// Verifies that a function's result generics are determined by its inputs.
///
/// Generics supplied by the caller, such as trait-level parameters on
/// a trait method, are treated as already determined. Every other generic in a
/// result must also appear in an input type, including claim inputs that encode
/// ordinary where-clause evidence. Otherwise function monomorphization has no
/// source of evidence for choosing that result type. This is intentionally a
/// syntactic check: the verifier does not try to invert equality predicates or
/// associated-type bindings to recover missing result generics.
static LogicalResult verifyFunctionResultGenericsAreDetermined(
    func::FuncOp function,
    const DenseSet<Type> &providedGenerics) {
  FunctionType functionType = function.getFunctionType();

  DenseSet<Type> inputGenerics;
  for (Type input : functionType.getInputs()) {
    auto generics = getGenericTypesIn(input);
    inputGenerics.insert(generics.begin(), generics.end());
  }

  DenseSet<Type> seenResultGenerics;
  SmallVector<GenericTypeInterface, 4> resultGenerics;
  for (Type result : functionType.getResults()) {
    for (auto generic : getGenericTypesIn(result)) {
      if (seenResultGenerics.insert(generic).second)
        resultGenerics.push_back(generic);
    }
  }

  for (Type resultGeneric : resultGenerics) {
    if (providedGenerics.contains(resultGeneric) || inputGenerics.contains(resultGeneric))
      continue;

    return function.emitOpError()
           << "function '" << function.getSymName()
           << "' result type contains type parameter " << resultGeneric
           << " that is not determined by any input type";
  }

  return success();
}

/// One local associated-type resolution rule available while normalizing a type.
///
/// The rule says that projections whose trait application is exactly `app` may
/// be resolved through `impl` after applying `subst` to the impl's associated
/// type binding. It represents evidence already present at the current IR
/// boundary; it does not perform global impl lookup.
struct LocalProjectionRule {
  ImplOp impl;
  TraitApplicationAttr app;
  SpecializationMap subst;
};

/// Context controlling how far `normalize` may resolve projection types.
///
/// The small core of normalization is deliberately local: callers add explicit
/// evidence-derived rules, and projection heads not justified by those rules are
/// preserved. Global resolver-backed normalization remains a separate lowering
/// concern.
class NormalizationContext {
public:
  void addLocalProjectionRule(ImplOp impl, TraitApplicationAttr app,
                              const SpecializationMap &subst) {
    localProjectionRules.push_back({impl, app, subst});
  }

  /// Resolves projections in `ty` using this context's local rules.
  ///
  /// The walk runs to a fixed point so a resolved associated type can expose
  /// another projection resolvable by the same local evidence.
  FailureOr<Type> normalize(
      Type ty,
      llvm::function_ref<InFlightDiagnostic()> err);

  /// Resolves projections in each input and result type of `functionType`.
  FailureOr<FunctionType> normalize(
      FunctionType functionType,
      llvm::function_ref<InFlightDiagnostic()> err);

private:
  SmallVector<LocalProjectionRule, 4> localProjectionRules;
};

} // namespace


//===----------------------------------------------------------------------===//
// NormalizationContext
//===----------------------------------------------------------------------===//

FailureOr<Type> NormalizationContext::normalize(
    Type ty,
    llvm::function_ref<InFlightDiagnostic()> err) {
  constexpr unsigned maxIterations = 64;

  auto normalizeOnce = [&](Type root) {
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](ProjectionType proj) -> std::optional<Type> {
      for (LocalProjectionRule &rule : localProjectionRules) {
        if (!rule.impl || proj.getTraitApplication() != rule.app)
          continue;

        auto resolved = rule.impl.specializeAssociatedTypeBinding(
            proj.getAssocName().getValue(), proj.getAssocTypeArgs());
        if (failed(resolved))
          continue;
        return rule.subst.apply(*resolved);
      }
      return std::nullopt;
    });
    return replacer.replace(root);
  };

  Type previous;
  for (unsigned i = 0; i != maxIterations; ++i) {
    previous = ty;
    ty = normalizeOnce(ty);
    if (ty == previous)
      return ty;
  }

  if (err)
    err() << "projection normalization did not converge; check for cyclic "
             "associated type bindings";
  return failure();
}

FailureOr<FunctionType> NormalizationContext::normalize(
    FunctionType functionType,
    llvm::function_ref<InFlightDiagnostic()> err) {
  auto normalized = normalize(Type(functionType), err);
  if (failed(normalized))
    return failure();
  return cast<FunctionType>(*normalized);
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

  // An endpoint mentions a type parameter when any generic hiding inside it is
  // one of the trait's parameters or a GAT parameter; getGenericTypesIn descends
  // projections (and opaque equality endpoints) that a plain walk would miss.
  auto endpointMentionsParam = [&](Type endpoint) {
    for (GenericTypeInterface g : getGenericTypesIn(endpoint))
      if (uniqueParams.contains(Type(g)) || gatParams.contains(Type(g)))
        return true;
    return false;
  };

  // check requirements
  for (Attribute pred : getRequirements()) {
    if (auto app = dyn_cast<TraitApplicationAttr>(pred)) {
      // each application requirement must use at least one of the trait's type
      // parameters OR at least one GAT type parameter
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
    } else if (auto eq = dyn_cast<TypeEqualityAttr>(pred)) {
      // An equality requirement has no trait head, so there is no
      // self-reference to forbid; it must still relate the trait's parameters,
      // mentioning at least one through either endpoint.
      if (!endpointMentionsParam(eq.getLhs()) &&
          !endpointMentionsParam(eq.getRhs()))
        return emitOpError() << "'where' clause equality requirement "
                             << ClaimType::getEquality(getContext(), eq)
                             << " must mention at least one type parameter";
    }
  }

  // check trait method result generics
  for (Operation &op : getBody().front()) {
    if (auto method = dyn_cast<func::FuncOp>(op)) {
      if (failed(verifyFunctionResultGenericsAreDetermined(method, uniqueParams)))
        return failure();
    }
  }

  return success();
}

LogicalResult TraitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // verify obligations
  return getRequirements().verifySymbolUses(getOperation(), symbolTable);
}

FailureOr<SpecializationMap> TraitOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                      llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();
  return buildSpecialization(getSelfClaim(), actualSelfClaim, *module, errFn);
}


SmallVector<ClaimType> TraitOp::getRequirementsAsClaims() {
  MLIRContext *ctx = getContext();
  // Each requirement becomes a claim of its arm: an application claim for an
  // application entry, an equality claim for an equality entry.
  SmallVector<ClaimType> result;
  for (Attribute pred : getRequirements()) {
    if (auto app = dyn_cast<TraitApplicationAttr>(pred))
      result.push_back(ClaimType::get(ctx, app));
    else if (auto eq = dyn_cast<TypeEqualityAttr>(pred))
      result.push_back(ClaimType::getEquality(ctx, eq));
  }
  return result;
}

FailureOr<SmallVector<ClaimType>> TraitOp::specializeRequirementsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // build a specialized substitution for actualSelfClaim
  auto spec = buildSpecialization(getSelfClaim(), actualSelfClaim, *module, errFn);
  if (failed(spec)) return failure();
  auto subst = spec->toTypeMap();

  // apply the substitution to each requirement
  return llvm::map_to_vector(getRequirementsAsClaims(), [&](ClaimType req) {
    ClaimType specializedReq = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(subst, req));
    if (!specializedReq)
      llvm_unreachable("TraitOp::specializeRequirementsAsClaimsFor: expected ClaimType");
    return specializedReq;
  });
}

SmallVector<ImplOp> TraitOp::getImpls() {
  auto module = getModule();
  if (failed(module)) return {};

  // Impls are top-level module children (ImplOp is HasParent<ModuleOp>), so scan
  // them directly and match this trait's symbol name. This avoids a full-module
  // symbol-use walk, which materializes every operation's attribute dictionary.
  StringRef traitName = getSymName();
  SmallVector<ImplOp> result;
  size_t scanned = 0;
  for (Operation &op : *module->getBody()) {
    ++scanned;
    auto impl = dyn_cast<ImplOp>(op);
    if (!impl)
      continue;
    TraitApplicationAttr selfApp = impl.getSelfApplication();
    if (selfApp && selfApp.getTraitName().getValue() == traitName)
      result.push_back(impl);
  }
  countCandidateScan(scanned);

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
  auto requirements = PredicateArrayAttr::get(ctx, ArrayRef<Attribute>());
  if (succeeded(p.parseOptionalKeyword("where"))) {
    requirements = dyn_cast_or_null<PredicateArrayAttr>(PredicateArrayAttr::parse(p,{}));
    if (!requirements)
      return p.emitError(p.getCurrentLocation(), "expected a predicate array");
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

/// A verified equality-armed witness in replayable form: the cited impl, the
/// projection's trait application, and the head-match substitution, ready for
/// NormalizationContext::addLocalProjectionRule.
struct ImplWitnessRule {
  ImplOp impl;
  TraitApplicationAttr app;
  SpecializationMap subst;
};

/// Verifies an impl's equality-armed witnesses and returns each as a local
/// resolution rule. Such a witness certifies that a sibling impl binds a
/// ground projection to a resolved type; a witness citing a conditional impl
/// is legal exactly when the impl's own where clause covers the cited impl's
/// assumptions or an application-armed witness supplies them. Each entry
/// verifies with an EMPTY equality modulus: sibling witnesses never serve as
/// each other's modulus, because an attribute array has no dominance and
/// mutual justification could ground a false equality on nothing. A witness
/// must name a GROUND projection -- resolving a poly-carrying projection by
/// unifying its variable with one cited impl's concrete head would accept a
/// generic impl on the strength of one instance.
static FailureOr<SmallVector<ImplWitnessRule>> collectImplWitnessRules(
    ImplOp impl, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  SmallVector<ImplWitnessRule> rules;
  ArrayAttr witnesses = impl.getWitnessesAttr();
  if (!witnesses)
    return rules;

  SmallVector<TraitApplicationAttr> obligationPremises(
      impl.getAssumptions().getApplications());
  // The application-armed witnesses cover a cited conditional impl's standing
  // assumptions; gather them first so every equality-armed witness below
  // verifies against the whole discharge set regardless of array order.
  SmallVector<WitnessAttr> dischargeWitnesses;
  for (Attribute entry : witnesses) {
    auto witness = cast<WitnessAttr>(entry);
    if (isa<TraitApplicationAttr>(witness.getPredicate()))
      dischargeWitnesses.push_back(witness);
  }
  for (Attribute entry : witnesses) {
    auto witness = cast<WitnessAttr>(entry);
    if (!isa<TypeEqualityAttr>(witness.getPredicate()))
      continue;
    auto projectionTy = dyn_cast<ProjectionType>(witness.getProjection());
    if (!projectionTy)
      return impl.emitOpError()
             << "a witness must name a projection, found "
             << witness.getProjection();
    if (isPolymorphicType(witness.getProjection()))
      return impl.emitOpError()
             << "witness projection " << witness.getProjection()
             << " is not ground; a witness resolves only a "
                "ground sibling projection";
    auto subst = verifyProjectionResolutionAtImpl(
        module, witness, /*premises=*/{}, obligationPremises,
        dischargeWitnesses, errFn);
    if (failed(subst))
      return failure();
    auto citedImpl = mlir::SymbolTable::lookupNearestSymbolFrom<ImplOp>(
        module, witness.getImplRef());
    rules.push_back(
        {citedImpl, projectionTy.getTraitApplication(), std::move(*subst)});
  }
  return rules;
}

static LogicalResult verifyEqualityObligations(
    ImplOp impl, TraitOp traitOp, ArrayRef<ImplWitnessRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn);

LogicalResult ImplOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&]{ return emitOpError(); };

  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // Verify self application attribute exists
  auto selfApp = getSelfApplication();
  if (!selfApp)
    return emitOpError() << "requires a self application TraitApplicationAttr";

  // Verify the self application
  if (failed(selfApp.verifySymbolUses(getOperation(), symbolTable)))
    return failure();

  // Verify the where-clause predicates' symbol uses: application entries name a
  // valid trait at the right arity; equality entries carry their symbol users
  // nested in the endpoints. The automatic symbol-user driver skips inherent
  // attributes, so the owning op delegates here, exactly as trait.trait does for
  // its requirements.
  if (failed(getAssumptions().verifySymbolUses(getOperation(), symbolTable)))
    return failure();

  // The verified witnesses become local resolution rules the comparisons below
  // replay after their own-binding rule; the fixed-point walk applies them
  // innermost-first, so a nested projection reduces its inner application
  // before its outer one.
  auto premiseRules = collectImplWitnessRules(*this, *module, errFn);
  if (failed(premiseRules))
    return failure();
  auto addPremiseRules = [&](NormalizationContext &ctx) {
    for (const ImplWitnessRule &r : *premiseRules)
      ctx.addLocalProjectionRule(r.impl, r.app, r.subst);
  };

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
      Type specializedTraitMethodTy = applySubstitutionToFixedPoint(traitSubst->toTypeMap(), traitMethodTy);

      NormalizationContext normalization;
      normalization.addLocalProjectionRule(
          *this, getSelfApplication(), *traitSubst);
      addPremiseRules(normalization);

      // Check that the impl method's signature can specialize to the expected
      // signature. Only projections naming this impl's exact trait application
      // may use its associated type bindings; unrelated projections with the
      // same associated type name remain symbolic.
      FunctionType implMethodTy = implMethod.getFunctionType();
      auto expectedMethodTy = normalization.normalize(
          specializedTraitMethodTy, errFn);
      if (failed(expectedMethodTy))
        return failure();
      auto actualMethodTy = normalization.normalize(implMethodTy, errFn);
      if (failed(actualMethodTy))
        return failure();

      // Substituting this impl's self application into the trait method
      // signature can mint a ground projection the impl's own bindings do
      // not resolve -- a sibling impl's application, e.g. Group[coop.block]::
      // Shape. A declared premise, verified and replayed above, reduces exactly
      // those projections, so both signatures reach this comparison at the same
      // grade. Compare their spellings with the module-free comparator: the
      // verifier enumerates no candidate impls. Any projection still standing is
      // not ground (the front end quotes an unresolved projection symbolically),
      // so it must match literally.
      if (failed(buildSpecialization(Type(*expectedMethodTy),
                                     Type(*actualMethodTy), ModuleOp(), errFn))) {
        return emitOpError() << "method '" << name << "' has incompatible signature: "
                             << "expected " << *expectedMethodTy
                             << " but found " << *actualMethodTy;
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

  if (failed(verifyEqualityObligations(*this, traitOp, *premiseRules, errFn)))
    return failure();

  return success();
}

/// Verifies each equality this impl must satisfy: the trait-header equality
/// requirements specialized for the impl's self arguments (e.g. Self::Output =
/// Self), and the equality entries of the impl's own where clause (e.g.
/// F::Output = Acc). Both endpoints normalize -- through the impl's own
/// bindings for its own application, or a declared witness rule for a sibling
/// -- and only a GROUND mismatch refuses. An endpoint that stays symbolic
/// cannot be decided here and is accepted; its correctness is established
/// where the impl is selected and, for evidence that is consumed, at the use
/// site (a false equality's coerce fails the erase barrier). Once a witness
/// rule reduces an endpoint to a ground value, a false equality refuses here
/// whether or not its evidence is ever consumed. Application requirements are
/// proved at impl selection, not here.
static LogicalResult verifyEqualityObligations(
    ImplOp impl, TraitOp traitOp, ArrayRef<ImplWitnessRule> witnessRules,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto selfEqNorm = [&](NormalizationContext &eqNorm) -> LogicalResult {
    auto subst = impl.buildSubstitutionForSelfClaim(impl.getSelfClaim(), errFn);
    if (failed(subst)) return failure();
    eqNorm.addLocalProjectionRule(impl, impl.getSelfApplication(), *subst);
    for (const ImplWitnessRule &r : witnessRules)
      eqNorm.addLocalProjectionRule(r.impl, r.app, r.subst);
    return success();
  };
  auto checkEquality =
      [&](NormalizationContext &eqNorm, TypeEqualityAttr eq,
          llvm::function_ref<InFlightDiagnostic()> mismatch) -> LogicalResult {
    auto lhsN = eqNorm.normalize(eq.getLhs(), errFn);
    if (failed(lhsN)) return failure();
    auto rhsN = eqNorm.normalize(eq.getRhs(), errFn);
    if (failed(rhsN)) return failure();
    if (isGroundType(*lhsN) && isGroundType(*rhsN) && *lhsN != *rhsN)
      return mismatch() << *lhsN << " and " << *rhsN << " are not the same type";
    return success();
  };

  // Each guard keeps the self-claim specialization off an impl with nothing of
  // that kind to check.
  bool hasEqualityRequirement = llvm::any_of(
      traitOp.getRequirements(), [](Attribute pred) {
        return mlir::isa<TypeEqualityAttr>(pred);
      });
  if (hasEqualityRequirement) {
    auto specReqs =
        traitOp.specializeRequirementsAsClaimsFor(impl.getSelfClaim(), errFn);
    if (failed(specReqs)) return failure();
    NormalizationContext eqNorm;
    if (failed(selfEqNorm(eqNorm))) return failure();
    for (ClaimType req : *specReqs) {
      auto eq = req.getEqualityAttr();
      if (!eq) continue;
      if (failed(checkEquality(eqNorm, eq, [&] {
            return impl.emitOpError()
                   << "does not satisfy trait-header equality requirement " << req
                   << ": ";
          })))
        return failure();
    }
  }

  bool assumesEquality = llvm::any_of(impl.getAssumptions(), [](Attribute pred) {
    return mlir::isa<TypeEqualityAttr>(pred);
  });
  if (assumesEquality) {
    NormalizationContext eqNorm;
    if (failed(selfEqNorm(eqNorm))) return failure();
    for (Attribute pred : impl.getAssumptions()) {
      auto eq = dyn_cast<TypeEqualityAttr>(pred);
      if (!eq) continue;
      if (failed(checkEquality(eqNorm, eq, [&] {
            return impl.emitOpError()
                   << "does not satisfy its own equality predicate " << eq << ": ";
          })))
        return failure();
    }
  }

  return success();
}

bool ImplOp::isUnconditional() {
  // an ImplOp is unconditional if:
  // 1. it is monomorphic (no type parameters),
  // 2. its TraitOp has no application requirements, and
  // 3. it assumes no application predicates.
  // An equality predicate -- whether a trait-header requirement or one of this
  // impl's own assumptions -- does not make an impl conditional: it is settled at
  // impl verification when its endpoints reduce to ground (through the impl's own bindings or
  // a declared premise) and deferred to selection and use otherwise, never proved
  // through impl selection. So only application predicates count against
  // unconditionality.
  return getTypeParams().empty() &&
         !getAssumptions().hasApplications() &&
         !getTrait().getRequirements().hasApplications();
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

FailureOr<SpecializationMap> ImplOp::buildSubstitutionForSelfClaim(ClaimType actualSelfClaim,
                                                                     llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto module = getModule(errFn);
  if (failed(module)) return failure();
  // Building this candidate impl's self-claim substitution is a computation
  // over its own committed facts, not the verifier's spelling comparison -- and
  // getCandidateImplsFor runs it as a per-candidate match probe before any impl
  // is chosen. Passing the module drives ground-projection resolution inside
  // unification, so the ground projections the match mints -- the actual
  // claim's arguments (a caller cast a witness to a projection spelling) and this
  // impl's own self application (a blanket impl spells `Trait[T]::A`, ground once
  // `T` binds) -- reduce to their determined values and the two meet at one
  // spelling. That settles only this candidate's own match; the rigid side is
  // never resolved, so a non-projection mismatch still fails.
  return buildSpecialization(getSelfClaim(), actualSelfClaim, *module, errFn);
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

FailureOr<ImplSpecialization> ImplOp::buildImplSpecialization(
    ClaimType provenSelfClaim,
    DemandOrigin origin,
    ProofDerivationMemo *memo,
    llvm::function_ref<InFlightDiagnostic()> err) {
  if (!provenSelfClaim.isProven()) {
    if (err) err() << "expected proven self claim for " << getSymName();
    return failure();
  }

  auto module = getModule(err);
  if (failed(module)) return failure();

  EvidenceBindings evidence;

  // Bind the same self claim without a proof to the proven self claim. This
  // recursively records claim -> proven-claim evidence bindings.
  ClaimType unprovenSelfClaim = provenSelfClaim.asUnproven();
  if (recordsToLedger(origin))
    countDerivationEntry(DerivationEntry::ImplSelfProof);
  if (failed(verifyAndRecordProof(unprovenSelfClaim, provenSelfClaim, *module,
                                  evidence, origin, memo, err)))
    return failure();

  auto specialization = buildSubstitutionForSelfClaim(provenSelfClaim, err);
  if (failed(specialization)) return failure();

  return ImplSpecialization(*specialization, evidence);
}

SmallVector<GenericTypeInterface, 4> ImplOp::getTypeParams() {
  // collect all the types where a type variable could hide
  SmallVector<Type> allOurTypes;
  allOurTypes.push_back(getSelfClaim());
  for (ClaimType a : getAssumptionsAsClaims()) {
    allOurTypes.push_back(a);
  }
  // An equality assumption's endpoints are opaque to the generic walk, so push
  // them directly: a generic that appears only in an assumed equality (e.g. the
  // accumulator in `F::Output = Acc`) is still one of this impl's parameters.
  for (Attribute pred : getAssumptions()) {
    if (auto eq = dyn_cast<TypeEqualityAttr>(pred)) {
      allOurTypes.push_back(eq.getLhs());
      allOurTypes.push_back(eq.getRhs());
    }
  }

  // tuple the types
  TupleType tupled = TupleType::get(getContext(), allOurTypes);

  // get all the generic types in the tuple
  return getGenericTypesIn(tupled);
}

FailureOr<func::FuncOp> ImplOp::getOrSpecializeMethod(OpBuilder& builder, StringRef methodName) {
  auto trait = getTrait();

  // check that we've named a valid trait method
  if (!trait.hasMethod(methodName)) return failure();

  // check if the method already exists in the ImplOp
  auto method = getMethod(methodName);
  if (succeeded(method)) return method;

  // otherwise, we need to specialize the method from the default implementation in the trait
  auto traitMethod = trait.getOptionalMethod(methodName);
  if (failed(traitMethod)) return failure();

  // build a substitution that maps trait PolyType parameters to impl type arguments
  auto subst = trait.buildSubstitutionForSelfClaim(getSelfClaim());
  if (failed(subst)) return failure();

  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&getBody().front());
  return specializePolymorph(builder, *traitMethod, methodName, subst->toTypeMap());
}

/// Specialize a polymorphic function and replace any AssumeOps whose
/// trait application matches a claim-typed function parameter.
static func::FuncOp specializeAndReplaceAssumes(
    PatternRewriter &rewriter, func::FuncOp callee,
    StringRef name, const DenseMap<Type,Type> &subst) {
  auto funcOp = specializePolymorph(rewriter, callee, name, subst);

  // A trait.assume materializes a hypothesis whose evidence, once the function
  // is specialized, is carried by a claim-typed parameter of the same arm: an
  // application assume is satisfied by an application parameter naming the same
  // trait application, an equality assume by an equality parameter carrying the
  // same equality.
  DenseMap<TraitApplicationAttr, Value> applicationParams;
  DenseMap<TypeEqualityAttr, Value> equalityParams;
  for (auto arg : funcOp.getArguments())
    if (auto claimTy = dyn_cast<ClaimType>(arg.getType())) {
      if (claimTy.isApplication())
        applicationParams[claimTy.getTraitApplication()] = arg;
      else if (auto eq = claimTy.getEqualityAttr())
        equalityParams[eq] = arg;
    }

  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    ClaimType claim = a.getClaim();
    Value replacement;
    if (auto eq = claim.getEqualityAttr()) {
      auto it = equalityParams.find(eq);
      if (it != equalityParams.end())
        replacement = it->second;
    } else {
      auto it = applicationParams.find(claim.getTraitApplication());
      if (it != applicationParams.end())
        replacement = it->second;
    }
    if (replacement) {
      rewriter.replaceAllUsesWith(a.getResult(), replacement);
      toErase.push_back(a);
    }
  });
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

static func::FuncOp specializeMethodAsFreeFuncWithLeadingSelfProof(
    PatternRewriter& rewriter,
    ModuleOp module,
    func::FuncOp method,
    StringRef functionName,
    ClaimType selfProofTy,
    const DenseMap<Type,Type>& subst) {

  // specialize the method into the grandparent with a mangled name
  PatternRewriter::InsertionGuard guard(rewriter);

  // clone the method into the method's grandparent
  rewriter.setInsertionPointAfter(method->getParentOp());

  // specialize the function and replace assumes matching claim-typed parameters
  auto funcOp = specializeAndReplaceAssumes(rewriter, method, functionName, subst);

  // prepend the self proof as the first parameter of the function and
  // set visibility to private
  rewriter.modifyOpInPlace(funcOp, [&] {
    (void)funcOp.insertArgument(/*idx=*/0, selfProofTy,
                               /*argAttrs=*/mlir::DictionaryAttr(),
                               method.getLoc());
    funcOp.setVisibility(SymbolTable::Visibility::Private);
  });
  BlockArgument selfProofArg = funcOp.getArgument(0);

  // replace remaining application AssumeOps with projections from selfProofArg.
  // A projection derives an application consequence of the self proof; an
  // equality is not a projection consequence (it is established by
  // trait.witness/coerce, not projected -- see ClaimType::getProjections), so an
  // equality assume is never rewritten to a projection here. Any equality assume
  // whose evidence is a claim-typed parameter was already discharged in
  // specializeAndReplaceAssumes above.
  SmallVector<AssumeOp> toErase;
  funcOp.walk([&](AssumeOp a) {
    if (a.getClaim().isEquality())
      return;

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(a);

    Value replacement = ProjectOp::create(
      rewriter,
      a.getLoc(),
      a.getClaim(),
      selfProofArg
    );

    rewriter.replaceAllUsesWith(a.getResult(), replacement);
    toErase.push_back(a);
  });

  // erase the AssumeOps
  for (auto a : toErase)
    rewriter.eraseOp(a);

  return funcOp;
}

FailureOr<func::FuncOp> ImplOp::getOrSpecializeFreeFunctionFromMethod(
    PatternRewriter& rewriter,
    ClaimType provenSelfClaim,
    StringRef methodName,
    const CallSubstitution &callSubst,
    ProofDerivationMemo *memo) {
  // check that methodName names a valid trait method
  if (!getTrait().hasMethod(methodName)) return failure();

  auto method = getOrSpecializeMethod(rewriter, methodName);
  if (failed(method)) return failure();

  auto implSpec =
      buildImplSpecialization(provenSelfClaim, DemandOrigin::ProofRecording,
                              memo);
  if (failed(implSpec)) return failure();

  // Build the same substitution that will be used to clone the method body:
  // first the enclosing impl substitution, then method-generic bindings from
  // this call site.
  DenseMap<Type,Type> subst = implSpec->toTypeMap();
  for (const auto &[k, v] : callSubst.toTypeMap())
    subst.try_emplace(k, v);

  // The extracted function name must include every substitution used to clone
  // the method body; otherwise different method-generic calls share a symbol.
  auto functionName = generateMangledName(provenSelfClaim) + "_" + methodName.str() +
    applySubstitutionAndGenerateMangledNameSuffix(subst, getGenericTypesIn((*method).getFunctionType()));

  MLIRContext* ctx = getContext();

  // look for an existing function
  auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
    getParentOp(),
    FlatSymbolRefAttr::get(ctx, functionName)
  );

  countCalleeSpecialization(/*cloned=*/!funcOp);
  if (!funcOp) {
    // specialize into grandparent with mangled name
    funcOp = specializeMethodAsFreeFuncWithLeadingSelfProof(
      rewriter,
      getParentOp(),
      *method,
      functionName,
      provenSelfClaim,
      subst
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
                                    PredicateArrayAttr assumptions) {
  // Build the full type-argument and where-clause signature for hashing. The
  // application entries hash exactly as an application-only impl always has, so
  // generalizing the where clause to carry equalities never perturbs the
  // synthesized name of an impl that assumes only applications. Equality entries
  // then contribute their own entropy, so two impls that differ only in an
  // equality assumption synthesize distinct names.
  std::string signature;
  llvm::raw_string_ostream os(signature);
  for (auto ty : selfApp.getTypeArgs()) {
    os << "_" << ty;
  }
  SmallVector<TraitApplicationAttr> apps =
      assumptions ? assumptions.getApplications()
                  : SmallVector<TraitApplicationAttr>{};
  if (!apps.empty()) {
    os << "_where";
    for (auto app : apps) {
      os << "_" << app.getTraitName().getValue();
      for (auto typeArg : app.getTypeArgs()) {
        os << "_" << typeArg;
      }
    }
  }
  if (assumptions) {
    bool firstEquality = true;
    for (Attribute pred : assumptions) {
      auto eq = dyn_cast<TypeEqualityAttr>(pred);
      if (!eq) continue;
      if (firstEquality) {
        os << "_eq";
        firstEquality = false;
      }
      os << "_" << eq.getLhs() << "_" << eq.getRhs();
    }
  }
  os.flush();

  return selfApp.getTraitName().getValue().str() + "_impl" + hashToSuffix(signature);
}

std::string ImplOp::generateMangledName(ClaimType claim) {
  auto subst = buildSubstitutionForSelfClaim(claim);
  if (failed(subst))
    llvm_unreachable("ImplOp::generateMangledName: specializedSelfClaimAgainst failed");

  return getSymName().str() + applySubstitutionAndGenerateMangledNameSuffix(*subst, getTypeParams());
}

SmallVector<ClaimType> ImplOp::getAssumptionsAsClaims() {
  MLIRContext *ctx = getContext();
  // The proof/derive/satisfiability streams read application-arm assumptions
  // only; equality entries are checked at impl verification against the impl's own bindings and
  // never proved through impl selection, so they are filtered out here at the
  // one place every obligation consumer flows through.
  return llvm::map_to_vector(getAssumptions().getApplications(),
                             [ctx](TraitApplicationAttr app) {
    return ClaimType::get(ctx, app);
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeAssumptionsAsClaimsFor(
    ClaimType actualSelfClaim,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // build a specialized substitution for actualSelfClaim. This runs while
  // partitioning candidates by their assumptions, before any impl is chosen, so
  // it settles only this one candidate's own match -- the self-claim build
  // reduces the ground projections it mints rather than crossing the tail.
  auto spec = buildSubstitutionForSelfClaim(actualSelfClaim, errFn);
  if (failed(spec)) return failure();
  auto subst = spec->toTypeMap();

  // apply the substitution to each assumption
  return llvm::map_to_vector(getAssumptionsAsClaims(), [&](ClaimType assumption) {
    ClaimType specializedAssumption = dyn_cast_or_null<ClaimType>(applySubstitutionToFixedPoint(subst, assumption));
    if (!specializedAssumption)
      llvm_unreachable("ImplOp::specializeAssumptionsAsClaimsFor: expected ClaimType");
    return specializedAssumption;
  });
}

FailureOr<SmallVector<ClaimType>> ImplOp::specializeObligationsAsClaimsFor(
    ClaimType actualSelfClaim,
    DemandOrigin origin,
    llvm::function_ref<InFlightDiagnostic()> errFn) {
  // specialize requirements of the trait
  auto requirements = getTrait().specializeRequirementsAsClaimsFor(actualSelfClaim, errFn);
  if (failed(requirements)) return failure();

  // The obligation stream is proved and derived through impl selection, an
  // application-arm operation. Trait-header equality requirements are checked
  // at impl verification against the impl's own bindings, never proved here, so they
  // do not enter the obligation stream (the proof/derive zips would have no
  // subproof for them).
  llvm::erase_if(*requirements, [](ClaimType c) { return c.isEquality(); });

  // Resolve projections in requirements using this impl's associated type
  // bindings (e.g., `Coord[Tensor[Self]::Shape]` becomes `Coord[tuple<i64,i64>]`
  // when the impl binds `Shape = S` and S is specialized to tuple<i64,i64>).
  // Only projections over this impl's own (actual) trait application resolve
  // through its bindings; a projection over a different trait application that
  // merely shares an associated-type name stays symbolic.
  auto subst = buildSubstitutionForSelfClaim(actualSelfClaim, errFn);
  if (failed(subst)) return failure();

  NormalizationContext normalization;
  normalization.addLocalProjectionRule(
      *this, actualSelfClaim.getTraitApplication(), *subst);
  for (ClaimType &req : *requirements) {
    auto resolved = normalization.normalize(req, errFn);
    if (failed(resolved)) return failure();
    req = cast<ClaimType>(*resolved);
    countObligationNormalization();

    // Normalization reads this impl's own bindings and nothing else, so a
    // projection over a sibling application survives it. The obligation goes on
    // to resolution spelled with that projection still standing, which is a
    // demand this normalization did not serve.
    if (DemandLedger::areObservationsEnabled())
      Type(req).walk([&](Type sub) {
        auto proj = dyn_cast<ProjectionType>(sub);
        if (proj && isMonomorphicType(proj))
          recordObligationNormalizationMiss(sub, origin);
      });
  }

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
  
  // where clause: one mixed PredicateArrayAttr (application and equality arms in
  // declaration order), stored directly as $assumptions -- no second array and
  // no partition. An application-only clause parses as a plain array of trait
  // applications.
  auto assumptions = PredicateArrayAttr::get(p.getContext(), ArrayRef<Attribute>{});
  if (succeeded(p.parseOptionalKeyword("where"))) {
    assumptions = dyn_cast_or_null<PredicateArrayAttr>(PredicateArrayAttr::parse(p, {}));
    if (!assumptions)
      return p.emitError(p.getCurrentLocation(), "expected a PredicateArrayAttr");
  }
  result.addAttribute("assumptions", assumptions);

  // Optional witnesses: one array of #trait.witness entries, each an
  // equality-armed projection-resolution witness or an application-armed
  // obligation discharge. Absent, the printed and parsed form is byte-identical
  // to an impl without them; the synthesized sym_name reads only the self
  // application and assumptions, so witnesses never perturb it.
  if (succeeded(p.parseOptionalKeyword("witnesses"))) {
    ArrayAttr witnesses;
    if (p.parseAttribute(witnesses))
      return failure();
    result.addAttribute("witnesses", witnesses);
  }

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

  // print witnesses if present and non-empty
  if (ArrayAttr witnesses = getWitnessesAttr()) {
    if (!witnesses.empty()) {
      printer << "witnesses ";
      printer.printAttribute(witnesses);
    }
  }

  printer.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"sym_name", "self_application", "assumptions", "witnesses"}
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

  // The proven claim is synthesized from the inherent trait_application
  // attribute, so it is not a type on this op's surface and the module-wide
  // type walk never verifies it. Verify the trait application here.
  if (failed(getTraitApplication().verifySymbolUses(getOperation(), symbolTable)))
    return failure();

  // check that the named impl exists
  auto implOp = getImpl();
  if (!implOp)
    return emitOpError() << "cannot find impl '" << getImplNameAttr() << "'";

  // check that we are able to substitute the impl's self claim against our proven claim
  if (failed(implOp.buildSubstitutionForSelfClaim(getProvenClaim(), errFn)))
    return failure();

  // recursively verify proof structure and that proof bindings can be recorded.
  // A verifier runs on whatever thread the verification was handed to and holds
  // no memo, so it derives what it needs itself.
  EvidenceBindings evidence;
  if (failed(verifyAndRecordProof(getProvenClaim().asUnproven(),
                                  getProvenClaim(), module, evidence,
                                  DemandOrigin::ProofVerification,
                                  /*memo=*/nullptr, errFn)))
    return failure();

  return success();
}

TraitOp ProofOp::getTrait() {
  auto module = getParentOp<ModuleOp>();
  if (!module)
    llvm_unreachable("ProofOp::getTrait: not inside a module");
  return getTraitApplication().getTraitOrAbort(module, "ProofOp::getTrait: couldn't find trait");
}

FailureOr<SmallVector<ClaimType>> ProofOp::verifyAndGetSubproofClaims(
    DemandOrigin origin, llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<ClaimType> result;

  ModuleOp module = getParentOp<ModuleOp>();
  if (!module) {
    if (err) err() << "not in a module";
    return failure();
  }

  // Compute obligations so we can validate coinductive self-references
  // and check arity.
  auto implOp = getImpl();
  if (!implOp) {
    if (err) err() << "cannot find impl '" << getImplNameAttr() << "'";
    return failure();
  }

  auto obligations =
      implOp.specializeObligationsAsClaimsFor(getProvenClaim(), origin, err);
  if (failed(obligations)) return failure();

  ArrayAttr subproofNames = getSubproofNames();
  if (subproofNames.size() != obligations->size()) {
    if (err) err() << "arity mismatch: expected " << obligations->size()
                   << " subproofs, but found " << subproofNames.size();
    return failure();
  }

  for (auto [obligation, name] : llvm::zip(*obligations, subproofNames)) {
    auto subproofRef = dyn_cast<FlatSymbolRefAttr>(name);
    if (!subproofRef) {
      if (err) err() << "expected FlatSymbolRefAttr";
      return failure();
    }

    // Coinductive self-reference: valid only when the obligation is for
    // the same trait as the proof itself.
    if (subproofRef.getValue() == getSymName()) {
      if (obligation.getTraitApplication().getTraitName() !=
              getTraitApplication().getTraitName()) {
        if (err) err() << "sub-proof '@" << subproofRef.getValue()
                       << "' must not reference the proof itself"
                       << " (proves " << getTraitApplication().getTraitName()
                       << " but obligation requires "
                       << obligation.getTraitApplication().getTraitName() << ")";
        return failure();
      }

      result.push_back(ClaimType::get(getContext(), getTraitApplication(), subproofRef));
      continue;
    }

    auto subproof = getProofOpOrUnconditionalImplOp(module, subproofRef, err);
    if (failed(subproof))
      return failure();

    TraitApplicationAttr subproofTraitApp;
    if (auto proofOp = dyn_cast<ProofOp>(*subproof))
      subproofTraitApp = proofOp.getTraitApplication();
    else
      subproofTraitApp = dyn_cast<ImplOp>(*subproof).getSelfApplication();

    result.push_back(ClaimType::get(getContext(), subproofTraitApp, subproofRef));
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
    llvm::function_ref<InFlightDiagnostic()> errFn,
    bool requireUnconditionalDirectImpl) {
  auto symOp = lookupProofSymbol(module, name, errFn);
  if (failed(symOp)) return failure();

  if (auto implOp = dyn_cast<ImplOp>(*symOp)) {
    if (requireUnconditionalDirectImpl &&
        failed(implOp.verifyIsUnconditional(errFn)))
      return failure();
    return implOp;
  }

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

// A spelled operand list: operands in parens, then their types in parens,
// `(%a, %b) : (T, U)`. SSA operands resolve against written types. The caller
// resolves the parsed operands once it knows where in the operand list they go.
static ParseResult parseTypedOperandList(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &types) {
  if (p.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      p.parseColon() ||
      p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&] {
        Type ty;
        if (p.parseType(ty))
          return failure();
        types.push_back(ty);
        return success();
      }))
    return failure();
  return success();
}

// Prints the `(%a, %b) : (T, U)` form parseTypedOperandList reads. The caller
// prints the keyword that precedes it.
static void printTypedOperandList(OpAsmPrinter &p, ValueRange operands) {
  p << "(";
  llvm::interleaveComma(operands, p, [&](Value v) { p.printOperand(v); });
  p << ") : (";
  llvm::interleaveComma(operands.getTypes(), p, [&](Type t) { p.printType(t); });
  p << ")";
}

ParseResult WitnessOp::parse(OpAsmParser &p, OperationState& result) {
  MLIRContext *ctx = p.getContext();

  // Equality proj-resolve arm: `proj_resolve !projection resolves !resolved
  // by @impl [given(%premises...) : (types...)] : <result-type>`.
  if (succeeded(p.parseOptionalKeyword("proj_resolve"))) {
    Type projection, resolved;
    FlatSymbolRefAttr citedImpl;
    if (p.parseType(projection) || p.parseKeyword("resolves") ||
        p.parseType(resolved) || p.parseKeyword("by") ||
        p.parseAttribute(citedImpl))
      return failure();
    auto err = [&] { return p.emitError(p.getCurrentLocation()); };
    auto equality = TypeEqualityAttr::getChecked(err, ctx, projection, resolved);
    if (!equality)
      return failure();
    auto witness = WitnessAttr::getChecked(err, ctx, Attribute(equality), citedImpl);
    if (!witness)
      return failure();
    result.addAttribute("witness", witness);

    if (succeeded(p.parseOptionalKeyword("given"))) {
      SmallVector<OpAsmParser::UnresolvedOperand> premises;
      SmallVector<Type> premiseTypes;
      if (parseTypedOperandList(p, premises, premiseTypes))
        return failure();
      if (p.resolveOperands(premises, premiseTypes, p.getCurrentLocation(),
                            result.operands))
        return failure();
    }

    Type resultType;
    if (p.parseColon() || p.parseType(resultType))
      return failure();
    result.addTypes(resultType);
    return success();
  }

  // Equality refl arm: `refl : <result-type>`.
  if (succeeded(p.parseOptionalKeyword("refl"))) {
    result.addAttribute("refl", UnitAttr::get(ctx));
    Type resultType;
    if (p.parseColon() || p.parseType(resultType))
      return failure();
    result.addTypes(resultType);
    return success();
  }

  // Equality composition arm: `compose(%premises...) : (types...) :
  // <result-type>`. The premise types are spelled -- SSA operands resolve
  // against written types -- and the result equality is spelled too, since it is
  // derived from the premises and not inferable from them.
  if (succeeded(p.parseOptionalKeyword("compose"))) {
    SmallVector<OpAsmParser::UnresolvedOperand> premises;
    SmallVector<Type> premiseTypes;
    if (parseTypedOperandList(p, premises, premiseTypes))
      return failure();
    if (p.resolveOperands(premises, premiseTypes, p.getCurrentLocation(),
                          result.operands))
      return failure();
    Type resultType;
    if (p.parseColon() || p.parseType(resultType))
      return failure();
    result.addTypes(resultType);
    return success();
  }

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
  if (auto witness = getWitnessAttr()) {
    p << " proj_resolve " << witness.getProjection() << " resolves "
      << witness.getResolved() << " by " << witness.getImplRef();
    if (!getPremises().empty()) {
      p << " given";
      printTypedOperandList(p, getPremises());
    }
    p << " : " << getResult().getType();
    return;
  }

  if (getRefl()) {
    p << " refl : " << getResult().getType();
    return;
  }

  // Composition arm: an equality result with neither a witness nor a refl
  // marker. Print the premises with their types and the spelled result equality.
  if (getResultClaim().isEquality()) {
    p << " compose";
    printTypedOperandList(p, getPremises());
    p << " : " << getResult().getType();
    return;
  }

  // Application arm.
  p << " " << getProofAttr() << " for ";
  getTraitApplicationAttr().print(p);

  p.printOptionalAttrDictWithKeyword(
    (*this)->getAttrs(),
    /*elidedAttrs=*/{"proof", "trait_application", "witness", "refl"}
  );
}

// The op's attributes must match the result claim's arm exactly, and the result
// type must equal the claim reconstructed from those attributes. For the
// equality arm, the current endpoints must be a single-substitution structural
// instance of the witness's endpoints (proj-resolve), identical (refl), or
// entailed by the premises' ground congruence closure (compose).
LogicalResult WitnessOp::verify() {
  ClaimType result = dyn_cast<ClaimType>(getResult().getType());
  if (!result)
    return emitOpError() << "result must be a !trait.claim";

  bool hasProof = static_cast<bool>(getProofAttr());
  bool hasApp = static_cast<bool>(getTraitApplicationAttr());
  bool hasWitness = static_cast<bool>(getWitnessAttr());
  bool hasRefl = getRefl();

  // Equality arm.
  if (result.isEquality()) {
    if (hasProof || hasApp)
      return emitOpError() << "an equality witness carries no proof or trait "
                              "application";
    if (hasWitness && hasRefl)
      return emitOpError() << "an equality witness carries at most one of a "
                              "proj-resolve leaf or a refl marker";
    TypeEqualityAttr eq = result.getEqualityAttr();

    if (hasRefl) {
      if (!getPremises().empty())
        return emitOpError() << "a refl witness takes no premises";
      if (eq.getLhs() != eq.getRhs())
        return emitOpError() << "a refl witness requires identical endpoints, "
                             << "found " << eq.getLhs() << " and " << eq.getRhs();
      return success();
    }

    if (hasWitness) {
      // proj-resolve: the current endpoints must be a single-substitution
      // structural instance of the witness's endpoints. The witness's
      // generic parameters are the variables; a single substitution must carry
      // the witness's projection and resolved type to the current pair. This
      // passes impl verification (identity), the clone-substituted state, and ground, and
      // rejects any non-substitution mangling. It is structural and local -- no
      // module lookup -- so the pair is matched with a null module.
      WitnessAttr witness = getWitnessAttr();
      // The witness slot carries a proj-resolve leaf, so its predicate is an
      // equality; a coerce discharge's application-headed witness has no place
      // here. Guard before reading the endpoints off the equality.
      if (!isa<TypeEqualityAttr>(witness.getPredicate()))
        return emitOpError() << "a proj-resolve witness must carry an "
                                "equality";
      MLIRContext *ctx = getContext();
      Type witnessPair = TupleType::get(ctx, {witness.getProjection(), witness.getResolved()});
      Type currentPair = TupleType::get(ctx, {eq.getLhs(), eq.getRhs()});
      if (failed(buildSpecialization(witnessPair, currentPair, ModuleOp())))
        return emitOpError() << "result endpoints " << eq.getLhs() << " = "
                             << eq.getRhs()
                             << " are not an instance of the witness "
                             << witness.getProjection() << " = " << witness.getResolved();
      return success();
    }

    // Composition: neither a witness nor refl. The result equality is
    // derived from the leaf equality premises by replaying the ground congruence
    // closure -- the transitivity and congruence that carry the premises to the
    // result are never stored, only the leaves are, so only definitional leaves
    // are ever stored. An equality claim carries no proof by that rule, so there
    // is no proof-swap for this arm to police.
    //
    // The composition arm is the only equality leaf whose evidence is another
    // claim value rather than a witness or an identical-endpoint marker, so
    // it is the only one whose validity can rest on its operands. In a region
    // without SSA dominance (a graph region such as a module body) a premise may
    // be the op's own result, letting two composes justify each other in a cycle
    // that grounds a false equality on nothing. Requiring an SSA-dominance region
    // makes the induction bottom out at proj-resolve- or refl-anchored leaves: a
    // false composition would need a false premise, which needs a false leaf, and
    // the proj-resolve and refl leaves refuse those.
    if (Region *parent = getOperation()->getParentRegion();
        parent && !mlir::mayHaveSSADominance(*parent))
      return emitOpError() << "a composition witness must be in a region that "
                              "enforces SSA dominance, so its premises cannot be "
                              "justified by its own result";
    if (getPremises().empty())
      return emitOpError() << "a composition witness requires at least one "
                              "equality premise";
    SmallVector<TypeEqualityAttr> premiseEqualities;
    for (Value premise : getPremises()) {
      auto claim = dyn_cast<ClaimType>(premise.getType());
      if (!claim || !claim.isEquality())
        return emitOpError() << "a composition witness premise must be an "
                                "equality claim, but a premise has type "
                             << premise.getType();
      premiseEqualities.push_back(claim.getEqualityAttr());
    }
    if (!entailedByGroundCongruence(eq.getLhs(), eq.getRhs(), premiseEqualities))
      return emitOpError() << "the premises do not entail " << eq.getLhs()
                           << " = " << eq.getRhs();
    return success();
  }

  // Application arm.
  if (hasWitness || hasRefl || !getPremises().empty())
    return emitOpError() << "an application witness carries neither a "
                            "proj-resolve leaf, a refl marker, nor premises";
  if (!hasProof || !hasApp)
    return emitOpError() << "an application witness carries a proof and a "
                            "trait application";
  if (result != getProvenClaim())
    return emitOpError() << "result " << result
                         << " does not match the witnessed claim "
                         << getProvenClaim();
  return success();
}

LogicalResult WitnessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitError() << "not inside a module";

  auto errFn = [&] { return emitOpError(); };

  // Equality proj-resolve arm: verify the citation where its symbol uses are
  // checked. The cited impl must bind the associated type the witness's
  // projection names to its resolved type, once specialized for the projection's
  // trait application, AND the witness's premises must discharge the cited impl's
  // own assumptions. The premises split by arm: equality claims are the
  // comparison modulus, application claims discharge the assumptions. The module
  // read runs here, on every full module verification -- not per consumer --
  // through the same obligation-aware check the C-API projection-resolution query
  // runs in obligation mode, so a consumer classifying a witness cannot
  // disagree with this verdict.
  if (auto witness = getWitnessAttr()) {
    SmallVector<TypeEqualityAttr> equalityPremises;
    SmallVector<TraitApplicationAttr> applicationPremises;
    for (Value premise : getPremises())
      if (auto claim = dyn_cast<ClaimType>(premise.getType())) {
        if (auto eq = claim.getEqualityAttr())
          equalityPremises.push_back(eq);
        else if (claim.isApplication())
          applicationPremises.push_back(claim.getTraitApplication());
      }
    return verifyProjectionResolutionAtUse(module, witness, equalityPremises,
                                           applicationPremises, errFn);
  }

  // Refl arm: nothing to verify here.
  if (getRefl())
    return success();

  // Composition arm: an equality result with neither a witness nor a refl
  // marker cites nothing by symbol -- its premises are SSA values -- so there is
  // no citation to verify here.
  if (getResultClaim().isEquality())
    return success();

  // Application arm: resolve the proof to its impl in one lookup; a
  // directly-named impl must be unconditional. The impl must then build a
  // substitution for our claim.
  auto impl = ProofOp::getImplFromProof(module, getProofAttr(), errFn,
                                        /*requireUnconditionalDirectImpl=*/true);
  if (failed(impl)) return failure();

  auto subst = impl->buildSubstitutionForSelfClaim(getProvenClaim(), errFn);
  return failed(subst) ? failure() : success();
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

  // A trait.derive discharges the cited impl's application-arm assumptions, so
  // every assumption operand must be a trait-application claim. An equality
  // claim carries no application to match and is not a legal derive operand.
  for (auto [i, operand] : llvm::enumerate(getAssumptions())) {
    auto operandClaim = cast<ClaimType>(operand.getType());
    if (!operandClaim.isApplication())
      return emitOpError() << "assumption operand #" << i << " (" << operandClaim
                           << ") must be a trait-application claim; an equality "
                              "claim is not a legal trait.derive operand";
  }

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
  MLIRContext *ctx = p.getContext();

  // `@Trait[...]` is an application hypothesis; `!A = !B` is an equality
  // hypothesis. The claim's result type wraps whichever predicate is parsed.
  FailureOr<Attribute> pred = parseApplicationOrEqualityPredicate(p);
  if (failed(pred))
    return failure();
  if (auto app = dyn_cast<TraitApplicationAttr>(*pred))
    st.addTypes(ClaimType::get(ctx, app));
  else
    st.addTypes(ClaimType::getEquality(ctx, cast<TypeEqualityAttr>(*pred)));

  return success();
}

void AssumeOp::print(OpAsmPrinter &p) {
  p << " ";

  ClaimType claim = getClaim();
  if (auto eq = claim.getEqualityAttr()) {
    // equality arm: `!lhs = !rhs`
    p << eq.getLhs() << " = " << eq.getRhs();
    return;
  }

  // application arm: print the assumed trait application
  claim.getTraitApplication().print(p);
}

LogicalResult AssumeOp::verify() {
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

  ClaimType claim = getClaim();
  TraitOp enclosingTrait = funcOp->getParentOfType<TraitOp>();
  ImplOp enclosingImpl = funcOp->getParentOfType<ImplOp>();

  // An assumed predicate is an axiom of the enclosing scope exactly when it
  // matches one by identity -- a method body shares the enclosing declaration's
  // polymorphic variables, so no weaker match is accepted. Application and
  // equality predicates are disjoint attribute kinds, so one set serves both
  // arms: an equality assume can match only an equality entry, an application
  // assume only an application entry. The sources are the enclosing function's
  // claim parameters, the enclosing impl's assumptions, the enclosing trait's
  // equality requirements, and -- anchoring an application assume as the impl's
  // assumption list anchors an equality one -- the enclosing trait's and impl's
  // own self-applications.
  DenseSet<Attribute> assumable;
  for (Type argType : funcOp.getArgumentTypes())
    if (auto c = dyn_cast<ClaimType>(argType)) {
      if (auto eq = c.getEqualityAttr())
        assumable.insert(eq);
      else if (c.isApplication())
        assumable.insert(c.getTraitApplication());
    }
  if (enclosingImpl) {
    assumable.insert(enclosingImpl.getSelfApplication());
    for (Attribute pred : enclosingImpl.getAssumptions())
      assumable.insert(pred);
  }
  if (enclosingTrait) {
    assumable.insert(enclosingTrait.getSelfApplication());
    for (Attribute pred : enclosingTrait.getRequirements())
      if (isa<TypeEqualityAttr>(pred))
        assumable.insert(pred);
  }

  if (auto assumedEq = claim.getEqualityAttr()) {
    if (!assumable.contains(assumedEq))
      return emitOpError() << "assumed equality " << assumedEq
                           << " is not assumable in this context";
    return success();
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
// CoerceOp
//===----------------------------------------------------------------------===//

ParseResult CoerceOp::parse(OpAsmParser &p, OperationState &st) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType, resultType;
  if (p.parseOperand(input) || p.parseColon() || p.parseType(inputType) ||
      p.parseKeyword("to") || p.parseType(resultType))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> equalities;
  SmallVector<Type> equalityTypes;
  if (succeeded(p.parseOptionalKeyword("via"))) {
    if (parseTypedOperandList(p, equalities, equalityTypes))
      return failure();
  }

  // The `unproven` marker is a trailing keyword: the printer emits no attribute
  // dictionary, so a declared, explicitly printed attribute is the only spelling
  // that survives a round trip.
  if (succeeded(p.parseOptionalKeyword("unproven")))
    st.addAttribute("unproven", p.getBuilder().getUnitAttr());

  st.addTypes(resultType);
  if (p.resolveOperand(input, inputType, st.operands))
    return failure();
  if (p.resolveOperands(equalities, equalityTypes, p.getCurrentLocation(),
                        st.operands))
    return failure();
  return success();
}

void CoerceOp::print(OpAsmPrinter &p) {
  p << " " << getInput() << " : " << getInput().getType() << " to "
    << getResult().getType();
  if (!getEqualities().empty()) {
    p << " via ";
    printTypedOperandList(p, getEqualities());
  }
  if (getUnproven())
    p << " unproven";
}

LogicalResult CoerceOp::verify() {
  // A verdict that is a pure function of op, operands, and attributes.

  // 1. Strip application-claim proofs from the input and result. Comparison
  // is modulo the proof, permanently.
  Type input = stripClaimProofs(getInput().getType());
  Type result = stripClaimProofs(getResult().getType());

  if (getUnproven()) {
    // The marked form cites nothing: its reconciling equalities are minted only
    // at monomorphization, so the endpoints stand in the pending judgment.
    if (!getEqualities().empty())
      return emitOpError()
             << "an unproven coerce may not cite equalities; it stands in a "
                "pending judgment discharged at monomorphization";
    if (failed(verifyPendingProjectionUnification(
            input, result, [&]() -> InFlightDiagnostic { return emitOpError(); })))
      return failure();
  } else {
    // 3. Collect the cited equalities; each operand must be an equality claim.
    SmallVector<TypeEqualityAttr> cited;
    for (Value e : getEqualities()) {
      auto claim = dyn_cast<ClaimType>(e.getType());
      if (!claim || !claim.isEquality())
        return emitOpError() << "coerce cites equality claims, but operand has "
                                "type " << e.getType();
      cited.push_back(claim.getEqualityAttr());
    }

    // 4. The two endpoints must fall in one class of the ground congruence
    // closure the cited equalities seed -- the shared entailment decision.
    if (!entailedByGroundCongruence(input, result, cited))
      return emitOpError() << "input type " << getInput().getType()
                           << " and result type " << getResult().getType()
                           << " are not equal under the cited equalities";
  }

  // 2. The no-proof-swap clause runs deep. The endpoints denote one claim once
  // the equalities reconcile them, so a proof present on the result and absent
  // or different on the input is a swap the coerce may not perform -- at every
  // position an application claim sits, not only the root. Positions are paired
  // by walking the two endpoint trees in lockstep off the same decomposition the
  // congruence closure keys on, over the unstripped types so the proofs are
  // still present.
  auto rejectProofSwap = [&](ClaimType fromClaim,
                             ClaimType toClaim) -> LogicalResult {
    if (!toClaim || !toClaim.isProven())
      return success();
    if (!fromClaim || !fromClaim.isProven() ||
        fromClaim.getProof() != toClaim.getProof())
      return emitOpError() << "may not swap the proof backing claim "
                           << toClaim.getTraitApplication()
                           << ": a coerce compares modulo a proof but may not "
                              "exchange it for another";
    return success();
  };
  // Does a proven application claim sit anywhere in this type?
  std::function<bool(Type)> carriesProvenClaim = [&](Type t) -> bool {
    if (auto c = dyn_cast<ClaimType>(t))
      if (c.isApplication() && c.isProven())
        return true;
    for (Type child : decomposeTerm(t).children)
      if (carriesProvenClaim(child))
        return true;
    return false;
  };
  std::function<LogicalResult(Type, Type)> checkNoSwap =
      [&](Type in, Type out) -> LogicalResult {
    if (failed(rejectProofSwap(dyn_cast<ClaimType>(in),
                               dyn_cast<ClaimType>(out))))
      return failure();
    TermShape di = decomposeTerm(in);
    TermShape dout = decomposeTerm(out);
    if (di.key == dout.key && di.children.size() == dout.children.size()) {
      for (auto [a, b] : llvm::zip(di.children, dout.children))
        if (failed(checkNoSwap(a, b)))
          return failure();
      return success();
    }
    // The two trees diverge in shape here, so no further positions pair. A proof
    // still standing on the result side has no input position to match and is a
    // swap; a proof-free divergence is the reconciliation the equalities
    // already licensed.
    if (carriesProvenClaim(out))
      return emitOpError() << "may not swap the proof backing a claim nested in "
                           << getResult().getType()
                           << ": a coerce compares modulo a proof but may not "
                              "exchange it for another";
    return success();
  };
  if (failed(checkNoSwap(getInput().getType(), getResult().getType())))
    return failure();

  return success();
}

OpFoldResult CoerceOp::fold(FoldAdaptor) {
  // The zero-evidence reflexive form is the discharged terminal state: it folds
  // to its operand, and any cited evidence then dies by ordinary DCE.
  if (getInput().getType() == getResult().getType())
    return getInput();
  return {};
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

  // A method call names its trait through the receiver claim's application, so
  // the receiver must be a trait-application claim. An equality claim names no
  // trait and is not a legal receiver.
  if (!claim.isApplication())
    return emitOpError() << "receiver (" << claim << ") must be a "
                            "trait-application claim; an equality claim names no "
                            "trait to call";

  // verify that the named trait matches the claim's trait
  auto expectedTraitAttr = getTraitAttr();
  auto foundTraitAttr = claim.getTraitApplication().getTraitName();
  if (expectedTraitAttr != foundTraitAttr)
    return emitOpError() << "expected claim for " << expectedTraitAttr << ", found " << foundTraitAttr;

  return success();
}

/// Adds projection normalization rules justified by one claim SSA value.
///
/// A rule records that projections for a specific trait application may use a
/// specific impl's associated type bindings while checking this operation.
static LogicalResult addLocalProjectionRulesFromClaim(
    NormalizationContext &ctx, Value claimValue, ModuleOp module,
    llvm::SmallPtrSetImpl<Operation *> &visitedDerives,
    llvm::function_ref<InFlightDiagnostic()> err) {
  // Only claim-typed operands can carry trait evidence relevant to projection
  // normalization. Ordinary method arguments do not contribute rules.
  auto claim = dyn_cast<ClaimType>(claimValue.getType());
  if (!claim)
    return success();

  // A proven claim names a proof symbol. That proof identifies the impl whose
  // associated type bindings justify reducing projections with this exact
  // trait application.
  if (claim.isProven()) {
    auto implOr = ProofOp::getImplFromProof(module, claim.getProof(), err);
    if (failed(implOr))
      return failure();

    // Store rules against the unproven application because projection heads do
    // not include proof symbols; proof only explains why the application holds.
    ClaimType unproven = claim.asUnproven();
    auto subst = implOr->buildSubstitutionForSelfClaim(unproven, err);
    if (failed(subst))
      return failure();

    ctx.addLocalProjectionRule(*implOr, unproven.getTraitApplication(), *subst);
    return success();
  }

  // A derive op also commits to one impl, but the evidence may be nested in its
  // assumptions. For example, a FnUni derive can carry the Fn claim that
  // resolves a closure's Output projection.
  if (auto derive = claimValue.getDefiningOp<DeriveOp>()) {
    // Derived claims can refer to other derived claims through assumptions; the
    // visited set keeps malformed or cyclic IR from recursing forever.
    if (!visitedDerives.insert(derive.getOperation()).second)
      return success();

    ImplOp impl = derive.getImplOp();
    if (!impl) {
      if (err)
        err() << "cannot find trait.impl '" << derive.getImplAttr() << "'";
      return failure();
    }

    ClaimType derived = derive.getDerivedClaim();
    auto subst = impl.buildSubstitutionForSelfClaim(derived, err);
    if (failed(subst))
      return failure();

    ctx.addLocalProjectionRule(impl, derived.getTraitApplication(), *subst);

    // Assumptions are part of the local evidence package used to derive this
    // claim, so their associated type bindings are visible to the same method
    // call comparison.
    for (Value assumption : derive.getAssumptions())
      if (failed(addLocalProjectionRulesFromClaim(
              ctx, assumption, module, visitedDerives, err)))
        return failure();
  }

  return success();
}

static FailureOr<NormalizationContext> buildLocalClaimNormalizationContext(
    ValueRange values, ModuleOp module,
    llvm::function_ref<InFlightDiagnostic()> err) {
  NormalizationContext ctx;
  llvm::SmallPtrSet<Operation *, 8> visitedDerives;
  for (Value value : values)
    if (failed(addLocalProjectionRulesFromClaim(
            ctx, value, module, visitedDerives, err)))
      return failure();
  return ctx;
}

LogicalResult MethodCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto errFn = [&]{ return emitOpError(); };

  auto module = getModule(errFn);
  if (failed(module)) return failure();

  // check that we can build a consistent substitution for this method call.
  // The verifier compares spellings with the module-free comparator: no
  // ground-projection resolution, so an unresolved crossing is a strict mismatch.
  return buildParameterSpecialization(/*unifyModule=*/ModuleOp(), errFn);
}

FailureOr<SpecializationMap> MethodCallOp::buildParameterSpecialization(ModuleOp unifyModule, llvm::function_ref<InFlightDiagnostic()> err) {
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
  Type formal = applySubstitutionToFixedPoint(traitSubst->toTypeMap(), *methodFormalTy);

  // solve the *call-site* specialization: unify the specialized formal type with the
  // actual call type to get any remaining bindings (including generics in args/results)
  FunctionType actual = getActualFunctionType();
  FunctionType originalActual = actual;

  // Specializing the method signature by the call claim can mint a ground
  // projection -- an argument or result that becomes a ground projection
  // meeting the caller's concrete spelling. Resolve every such projection on BOTH
  // sides BEFORE the input specialization, so the strict comparison meets two
  // spellings reduced to one grade rather than tolerating the crossing (the
  // input unification below runs before the local-claim normalization, so a
  // projection left here reaches the tail there).
  //
  // The call claim's evidence is a LOCALITY LICENSE, not a filter: it grants
  // permission to consult module facts at this call. Once granted, resolution
  // is claim-INDEPENDENT -- every ground projection whose application a unique
  // module impl binds is read to its determined value, whether or not this claim
  // covers it. So the gate does not itself prevent laundering; safety for a
  // projection the evidence does not cover rests on the trusted-producer boundary:
  // the front end discharges a ground projection's head claim where it spells
  // the projection, before this op sees it. An ordinary unproven claim withholds
  // the license entirely -- with no committing evidence at the site this op
  // reads no module facts, so its ground projections stay spelled and the strict
  // comparison rejects a head mismatch (the
  // invalid_method_call_unproven_projection_mismatch pin). Projections over the
  // method's own generics stay spelled (still polymorphic) for the local-claim
  // normalization; originalActual keeps the pre-resolution spelling for the
  // proof-binding recording below.
  //
  // `unifyModule` says which caller this is: a verifier passes none, a pass
  // passes the module. The boundary resolution below reads module facts either
  // way, so the demand it raises is classified by that same discriminator.
  ClaimType callClaim = getClaimType();
  bool callClaimHasEvidence =
      callClaim.isProven() || getClaim().getDefiningOp<DeriveOp>();
  DemandOrigin origin = unifyModule ? DemandOrigin::CallSiteSpecialization
                                    : DemandOrigin::CallSignatureVerification;
  if (callClaimHasEvidence) {
    formal = resolveGroundProjectionsByLookup(formal, *module, origin);
    actual = cast<FunctionType>(
        resolveGroundProjectionsByLookup(actual, *module, origin));
  } else {
    // No evidence, no license: this call never asked what its ground projections
    // resolve to, so those demands reach no engine at all. Nothing records
    // them either: method-call lowering, the only in-stage caller of this
    // specialization, defers until the call's claim is proven, and a proven
    // claim carries the license -- so every withheld call is a verifier's, and
    // a verifier's demand is counted rather than entered in a ledger. The
    // statistic is the whole of what this site can say.
    countWithheldCallClaim();
  }

  SmallVector<Value> localClaims;
  localClaims.push_back(getClaim());
  for (Value argument : getArguments())
    if (isa<ClaimType>(argument.getType()))
      localClaims.push_back(argument);

  auto normalization = buildLocalClaimNormalizationContext(localClaims, *module, err);
  if (failed(normalization))
    return failure();

  // The input and result specializations compare spellings that the boundary
  // resolution and the local-claim normalization above have already reduced.
  // `unifyModule` selects the comparator: a verifier passes none (strict), a
  // pass passes the module so binding a generic mid-solve resolves the ground
  // projection it mints.
  FunctionType formalFunction = cast<FunctionType>(formal);
  auto inputSpec = buildSpecialization(
      FunctionType::get(getContext(), formalFunction.getInputs(), TypeRange{}),
      FunctionType::get(getContext(), actual.getInputs(), TypeRange{}),
      unifyModule, err);
  if (failed(inputSpec)) return failure();

  // Input arguments determine method generics such as the closure type `F`.
  // Normalize only after those bindings are applied, so projections like
  // `Fn[F, Args]::Output` can match local evidence for the actual closure.
  auto inputMap = inputSpec->toTypeMap();
  auto normalizedFormal = normalization->normalize(
      cast<FunctionType>(applySubstitutionToFixedPoint(inputMap, formal)),
      err);
  if (failed(normalizedFormal))
    return failure();
  auto normalizedActual = normalization->normalize(
      cast<FunctionType>(applySubstitutionToFixedPoint(inputMap, actual)),
      err);
  if (failed(normalizedActual))
    return failure();
  formal = *normalizedFormal;
  actual = *normalizedActual;

  auto resultSpec = buildSpecialization(formal, actual, unifyModule, err);
  if (failed(resultSpec)) return failure();

  auto merged = inputMap;
  for (auto [key, value] : resultSpec->toTypeMap()) {
    auto existing = merged.find(key);
    if (existing != merged.end() && existing->second != value) {
      if (err)
        err() << "inconsistent method-call specialization for " << key
              << ": " << existing->second << " versus " << value;
      return failure();
    }
    merged[key] = value;
  }
  normalizeSubstitutionInPlace(merged);

  // The proofs this call's actual signature spells are bound where the call is
  // lowered: the factory that closes the substitution walks the same spellings
  // and reads them off the record. A verifier has no lowering behind it, so it
  // checks them here or nowhere, and it runs on a worker thread with no memo of
  // its own to serve them from.
  if (!unifyModule) {
    EvidenceBindings evidence;
    if (failed(bindProofsIn(originalActual, *module, evidence, origin,
                                     /*memo=*/nullptr, err)))
      return failure();
  }

  return SpecializationMap::fromTypeMap(merged);
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

FailureOr<func::FuncOp> MethodCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter,
    const CallSubstitution &subst,
    ProofDerivationMemo *memo) {
  ClaimType claimTy = cast<ClaimType>(getClaim().getType());
  return getProvenImpl()
    .getOrSpecializeFreeFunctionFromMethod(rewriter, claimTy, getMethodName(),
                                           subst, memo);
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

  // check that we can build a substitution. The verifier compares spellings
  // with the module-free comparator (no ground-projection resolution).
  return buildParameterSpecialization(/*unifyModule=*/ModuleOp(), errFn);
}

FailureOr<SpecializationMap> FuncCallOp::buildParameterSpecialization(ModuleOp unifyModule, llvm::function_ref<InFlightDiagnostic()> err) {
  auto module = getModule(err);
  if (failed(module)) return failure();

  // get formal and actual function types
  auto maybeFormal = getCalleeFunctionType(err);
  if (failed(maybeFormal)) return failure();

  FunctionType formal = *maybeFormal;
  FunctionType actual = getActualFunctionType();

  // build a substitution unifying formal & actual. `unifyModule` selects the
  // comparator: a verifier passes none (strict); a pass passes the module so a
  // ground projection minted by binding a generic mid-solve resolves.
  auto spec = buildSpecialization(formal, actual, unifyModule, err);
  if (failed(spec)) return failure();

  // `unifyModule` says which caller this is, the same discriminator the
  // boundary resolution above reads: a verifier passes none, a pass passes the
  // module.
  DemandOrigin origin = unifyModule
                            ? DemandOrigin::CallSiteSpecialization
                            : DemandOrigin::CallSignatureVerification;
  // The proofs this call's actual signature spells are bound where the call is
  // lowered: the factory that closes the substitution walks the same spellings
  // and reads them off the record. A verifier has no lowering behind it, so it
  // checks them here or nowhere, and it runs on a worker thread with no memo of
  // its own to serve them from.
  if (!unifyModule) {
    EvidenceBindings evidence;
    if (failed(bindProofsIn(actual, *module, evidence, origin,
                                     /*memo=*/nullptr, err)))
      return failure();
  }

  return *spec;
}

/// The name the instance of `op`'s callee carries, given the substitution that
/// specializes its body.
///
/// Mangling reads the specialization map alone, which is written when the
/// substitution is built and is not touched by closing it -- closing adds
/// projection and evidence bindings -- so the name a call is wired to and the
/// body it is wired to are read off one object.
static std::string calleeInstanceName(FuncCallOp op,
                                      const CallSubstitution &subst) {
  return op.getCalleeName().str() +
         applySubstitutionAndGenerateMangledNameSuffix(subst.getSpecialization(),
                                                       op.getCalleeTypeParams());
}

FailureOr<func::FuncOp> FuncCallOp::getOrSpecializeCallee(
    PatternRewriter &rewriter,
    const CallSubstitution &subst,
    ProofDerivationMemo *memo) {
  auto module = getModule();
  if (failed(module)) return failure();

  std::string instanceName = calleeInstanceName(*this, subst);
  auto *symOp = SymbolTable::lookupSymbolIn(*module, rewriter.getStringAttr(instanceName));
  func::FuncOp existing = dyn_cast_or_null<func::FuncOp>(symOp);
  if (existing) {
    countCalleeSpecialization(/*cloned=*/false);
    return existing;
  }

  auto callee = getCallee();
  if (failed(callee)) return failure();

  countCalleeSpecialization(/*cloned=*/true);
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(*callee);
  return specializeAndReplaceAssumes(rewriter, *callee, instanceName, subst.toTypeMap());
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

  // The result is either an application projection (@DstTrait[...] (by
  // @DstProof)?) or the equality hop to a trait's equality requirement
  // (!A = !B), disambiguated by the leading `@`. The equality arm never carries
  // a proof.
  ClaimType dstTy;
  FlatSymbolRefAttr dstTrait;
  OptionalParseResult dstSym = p.parseOptionalAttribute(dstTrait);
  if (dstSym.has_value()) {
    if (failed(*dstSym))
      return failure();
    if (p.parseLSquare())
      return failure();
    SmallVector<Type> dstArgs;
    do {
      Type ty;
      if (p.parseType(ty))
        return failure();
      dstArgs.push_back(ty);
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRSquare())
      return failure();
    auto dstApp = TraitApplicationAttr::get(p.getContext(), dstTrait,
                                            ArrayRef<Type>(dstArgs));

    // (by @DstProof)?
    FlatSymbolRefAttr dstProof;
    if (succeeded(p.parseOptionalKeyword("by"))) {
      if (p.parseAttribute(dstProof))
        return failure();
    }
    dstTy = dstProof ? ClaimType::get(p.getContext(), dstApp, dstProof)
                     : ClaimType::get(p.getContext(), dstApp);
  } else {
    Type lhs, rhs;
    if (p.parseType(lhs) || p.parseEqual() || p.parseType(rhs))
      return failure();
    dstTy = ClaimType::getEquality(p.getContext(), lhs, rhs);
  }
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

  // Destination: to @DstTrait[...] (by @DstProof)? or the equality hop to !A = !B
  p << " to ";
  ClaimType dstTy = getResultClaim();
  if (auto eq = dstTy.getEqualityAttr()) {
    eq.print(p);
  } else {
    dstTy.getTraitApplication().print(p);
    if (dstTy.isProven())
      p << " by " << dstTy.getProof();
  }
}

LogicalResult ProjectOp::verifySymbolUses(SymbolTableCollection &/*symbolTable*/) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError() << "not in a module";

  ClaimType src = getSourceClaim();
  ClaimType dst = getResultClaim();

  // Verify proofness parity for an application result: a proven source projects
  // to a proven result, an unproven to an unproven. An equality result is
  // exempt -- an equality claim is never proven, so projecting one from a proven
  // source does not force a proof it cannot carry.
  if (!dst.isEquality()) {
    bool srcProven = src.isProven();
    bool dstProven = dst.isProven();
    if (srcProven != dstProven) {
      if (!srcProven)
        return emitOpError() << "result cannot have 'by' when source has no 'by'";
      return emitOpError() << "result must have 'by' when source has 'by'";
    }
  }

  // The result must be one of the source's candidate projections.
  if (src.projectsTo(module, dst))
    return success();

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

  // parse optional `unsafe` keyword
  UnitAttr unsafeAttr;
  if (succeeded(p.parseOptionalKeyword("unsafe")))
    unsafeAttr = p.getBuilder().getUnitAttr();
  if (unsafeAttr)
    st.addAttribute("unsafe", unsafeAttr);

  // result type is the claim of the trait application
  auto claimTy = ClaimType::get(p.getContext(), app);
  st.addTypes(claimTy);

  return success();
}

void AllegeOp::print(OpAsmPrinter &p) {
  p << " ";

  // print the claimed trait application
  getClaim().getTraitApplication().print(p);

  // print optional unsafe
  if (getUnsafe())
    p << " unsafe";
}

LogicalResult AllegeOp::verify() {
  // claim must be monomorphic unless unsafe
  if (!getUnsafe() && !getClaim().isMonomorphic())
    return emitOpError() << "expected monomorphic claim, got "
                         << getClaim();
  return success();
}
