// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "c_api.h"
#include "Passes.hpp"
#include "Trait.hpp"
#include "TraitAttributes.hpp"
#include "TraitOps.hpp"
#include "TraitTypes.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::trait;

extern "C" {

void traitRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<TraitDialect>();
}

MlirPass traitCreateMonomorphizePass() {
  return wrap(createMonomorphizePass().release());
}

MlirPass traitCreateInstantiateMonomorphsPass() {
  return wrap(createInstantiateMonomorphsPass().release());
}

MlirPass traitCreateErasePolymorphsPass() {
  return wrap(createErasePolymorphsPass().release());
}

MlirAttribute traitTraitApplicationAttrGet(MlirContext wrappedCtx,
                                           MlirStringRef traitName,
                                           MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  OpBuilder builder(ctx);

  SmallVector<Attribute> typeAttrs;
  typeAttrs.reserve(numTypeArgs);
  for (intptr_t i = 0; i < numTypeArgs; ++i)
    typeAttrs.push_back(TypeAttr::get(unwrap(typeArgs[i])));

  auto traitRef = FlatSymbolRefAttr::get(
    ctx, StringRef(traitName.data, traitName.length)
  );
  auto typeArgsAttr = builder.getArrayAttr(typeAttrs);

  return wrap(TraitApplicationAttr::get(ctx, traitRef, typeArgsAttr));
}

bool traitAttributeIsATraitApplication(MlirAttribute attribute) {
  return isa<TraitApplicationAttr>(unwrap(attribute));
}

MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* wrappedTypeParams, intptr_t numTypeParams,
                                 MlirAttribute* requirements, intptr_t numRequirements) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Type> typeParams;
  typeParams.reserve(numTypeParams);
  for (intptr_t i = 0; i < numTypeParams; ++i) {
    typeParams.push_back(unwrap(wrappedTypeParams[i]));
  }

  SmallVector<TraitApplicationAttr> appAttrs;
  appAttrs.reserve(numRequirements);
  for (intptr_t i = 0; i < numRequirements; ++i) {
    auto app = dyn_cast<TraitApplicationAttr>(unwrap(requirements[i]));
    if (!app) return {}; // invalid type of attribute
    appAttrs.push_back(app);
  }

  auto op = TraitOp::create(builder,
    unwrap(loc),
    builder.getStringAttr(StringRef(name.data, name.length)),
    typeParams,
    appAttrs
  );

  return wrap(op.getOperation());
}

MlirOperation traitTraitOpCreateWithPredicates(MlirLocation loc, MlirStringRef name,
                                               MlirType* wrappedTypeParams, intptr_t numTypeParams,
                                               MlirAttribute* predicates, intptr_t numPredicates) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Type> typeParams;
  typeParams.reserve(numTypeParams);
  for (intptr_t i = 0; i < numTypeParams; ++i) {
    typeParams.push_back(unwrap(wrappedTypeParams[i]));
  }

  // The mixed where-clause: each predicate is a trait application or a type
  // equality. A non-predicate attribute is rejected by returning a null op.
  SmallVector<Attribute> preds;
  preds.reserve(numPredicates);
  for (intptr_t i = 0; i < numPredicates; ++i) {
    Attribute p = unwrap(predicates[i]);
    if (!isa<TraitApplicationAttr, TypeEqualityAttr>(p))
      return {};
    preds.push_back(p);
  }
  auto predsAttr = PredicateArrayAttr::get(ctx, preds);

  auto op = TraitOp::create(builder,
    unwrap(loc),
    builder.getStringAttr(StringRef(name.data, name.length)),
    typeParams,
    predsAttr
  );

  return wrap(op.getOperation());
}

MlirOperation traitImplOpCreate(MlirLocation loc,
                                MlirAttribute wrappedSelfTraitApp,
                                MlirAttribute* assumptions, intptr_t numAssumptions) {
  TraitApplicationAttr selfApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedSelfTraitApp));
  if (!selfApp) return {}; // invalid type of attribute

  SmallVector<TraitApplicationAttr> appAttrs;
  for (intptr_t i = 0; i < numAssumptions; ++i) {
    auto app = dyn_cast<TraitApplicationAttr>(unwrap(assumptions[i]));
    if (!app) return {}; // invalid type of attribute
    appAttrs.push_back(app);
  }

  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto op = ImplOp::create(builder,
    unwrap(loc),
    selfApp,
    appAttrs
  );

  return wrap(op.getOperation());
}

MlirOperation traitImplOpCreateNamed(MlirLocation loc,
                                     MlirStringRef symName,
                                     MlirAttribute wrappedSelfTraitApp,
                                     MlirAttribute* assumptions, intptr_t numAssumptions) {
  TraitApplicationAttr selfApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedSelfTraitApp));
  if (!selfApp) return {}; // invalid type of attribute

  SmallVector<TraitApplicationAttr> appAttrs;
  for (intptr_t i = 0; i < numAssumptions; ++i) {
    auto app = dyn_cast<TraitApplicationAttr>(unwrap(assumptions[i]));
    if (!app) return {}; // invalid type of attribute
    appAttrs.push_back(app);
  }

  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto op = ImplOp::create(builder,
    unwrap(loc),
    StringRef(symName.data, symName.length),
    selfApp,
    appAttrs
  );

  return wrap(op.getOperation());
}

MlirOperation traitImplOpCreateNamedWithPredicates(MlirLocation loc,
                                                   MlirStringRef symName,
                                                   MlirAttribute wrappedSelfTraitApp,
                                                   MlirAttribute* predicates, intptr_t numPredicates) {
  TraitApplicationAttr selfApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedSelfTraitApp));
  if (!selfApp) return {}; // invalid type of attribute

  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  // The impl's mixed where-clause: application entries are proof obligations,
  // equality entries assert the impl's own bindings. A non-predicate attribute
  // is rejected by returning a null op.
  SmallVector<Attribute> preds;
  preds.reserve(numPredicates);
  for (intptr_t i = 0; i < numPredicates; ++i) {
    Attribute p = unwrap(predicates[i]);
    if (!isa<TraitApplicationAttr, TypeEqualityAttr>(p))
      return {};
    preds.push_back(p);
  }
  auto predsAttr = PredicateArrayAttr::get(ctx, preds);

  auto op = ImplOp::create(builder,
    unwrap(loc),
    StringRef(symName.data, symName.length),
    selfApp,
    predsAttr
  );

  return wrap(op.getOperation());
}

MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName,
                                      MlirStringRef methodName,
                                      MlirValue claim,
                                      MlirValue* arguments, intptr_t numArguments,
                                      MlirType* resultTypes, intptr_t numResults) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> args;
  for (intptr_t i = 0; i < numArguments; ++i) {
    args.push_back(unwrap(arguments[i]));
  }

  SmallVector<Type> results;
  for (intptr_t i = 0; i < numResults; ++i) {
    results.push_back(unwrap(resultTypes[i]));
  }

  auto op = MethodCallOp::create(builder,
    unwrap(loc),
    results,
    StringRef(traitName.data, traitName.length),
    StringRef(methodName.data, methodName.length),
    unwrap(claim),
    args
  );

  return wrap(op.getOperation());
}

MlirOperation traitFuncCallOpCreate(MlirLocation loc,
                                    MlirStringRef callee,
                                    MlirValue* arguments, intptr_t numArguments,
                                    MlirType* resultTypes, intptr_t numResults) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> args;
  for (intptr_t i = 0; i < numArguments; ++i) {
    args.push_back(unwrap(arguments[i]));
  }

  SmallVector<Type> results;
  for (intptr_t i = 0; i < numResults; ++i) {
    results.push_back(unwrap(resultTypes[i]));
  }

  auto op = FuncCallOp::create(builder,
    unwrap(loc),
    results,
    FlatSymbolRefAttr::get(ctx, StringRef(callee.data, callee.length)),
    args
  );

  return wrap(op.getOperation());
}

MlirOperation traitAllegeOpCreate(MlirLocation loc,
                                  MlirAttribute wrappedTraitApp) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  OpBuilder builder(ctx);
  auto op = AllegeOp::create(builder,
    unwrap(loc),
    traitApp
  );

  return wrap(op.getOperation());
}

MlirOperation traitAllegeUnsafeOpCreate(MlirLocation loc,
                                        MlirAttribute wrappedTraitApp) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  OpBuilder builder(ctx);
  auto op = AllegeOp::create(builder,
    unwrap(loc),
    traitApp,
    /*isUnsafe=*/true
  );

  return wrap(op.getOperation());
}

MlirOperation traitWitnessOpCreate(MlirLocation loc,
                                   MlirStringRef proofName,
                                   MlirAttribute wrappedTraitApp) {
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  FlatSymbolRefAttr proofRef = FlatSymbolRefAttr::get(ctx, StringRef(proofName.data, proofName.length));

  auto op = WitnessOp::create(builder,
    unwrap(loc),
    proofRef,
    traitApp
  );

  return wrap(op.getOperation());
}

MlirOperation traitProofOpCreate(MlirLocation loc,
                                 MlirStringRef symName,
                                 MlirStringRef implName,
                                 MlirAttribute wrappedTraitApp,
                                 MlirStringRef* subproofNames, intptr_t numSubproofs) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  SmallVector<FlatSymbolRefAttr> subproofRefs;
  subproofRefs.reserve(numSubproofs);
  for (intptr_t i = 0; i < numSubproofs; ++i) {
    subproofRefs.push_back(
      FlatSymbolRefAttr::get(ctx, StringRef(subproofNames[i].data, subproofNames[i].length))
    );
  }

  OpBuilder builder(ctx);
  auto op = ProofOp::create(builder,
    unwrap(loc),
    StringRef(symName.data, symName.length),
    FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length)),
    traitApp,
    subproofRefs
  );

  return wrap(op.getOperation());
}

MlirOperation traitProjectOpCreate(MlirLocation loc,
                                   MlirValue srcClaim,
                                   MlirAttribute destTraitApp) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  MlirType wrappedResultType = traitClaimTypeGet(wrap(ctx), destTraitApp);
  ClaimType resultType = dyn_cast<ClaimType>(unwrap(wrappedResultType));
  if (!resultType) return {}; // invalid result type

  OpBuilder builder(ctx);

  auto op = ProjectOp::create(builder,
    unwrap(loc),
    resultType,
    unwrap(srcClaim)
  );

  return wrap(op.getOperation());
}

MlirOperation traitDeriveOpCreate(MlirLocation loc,
                                  MlirAttribute wrappedTraitApp,
                                  MlirStringRef implName,
                                  MlirValue* assumptions, intptr_t numAssumptions) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  FlatSymbolRefAttr implRef = FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length));

  SmallVector<Value> args;
  args.reserve(numAssumptions);
  for (intptr_t i = 0; i < numAssumptions; ++i)
    args.push_back(unwrap(assumptions[i]));

  OpBuilder builder(ctx);
  auto op = DeriveOp::create(builder,
    unwrap(loc),
    traitApp,
    implRef,
    args
  );

  return wrap(op.getOperation());
}

MlirOperation traitAssumeOpCreate(MlirLocation loc,
                                  MlirAttribute wrappedTraitApp) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  OpBuilder builder(ctx);

  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  auto op = AssumeOp::create(builder, unwrap(loc), traitApp);

  return wrap(op.getOperation());
}

MlirOperation traitAssumeOpCreateEquality(MlirLocation loc,
                                          MlirType lhs, MlirType rhs) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto eq = TypeEqualityAttr::getChecked(
      [&] { return emitError(unwrap(loc)); }, ctx, unwrap(lhs), unwrap(rhs));
  if (!eq)
    return {}; // endpoints carry a proof receipt, or are otherwise ill-formed
  auto claimTy = ClaimType::getEquality(ctx, eq);

  auto op = AssumeOp::create(builder, unwrap(loc), claimTy);

  return wrap(op.getOperation());
}

MlirType traitPolyTypeGet(MlirContext wrappedCtx,
                          unsigned int uniqueId) {
  return wrap(PolyType::get(unwrap(wrappedCtx), uniqueId));
}

MlirType traitClaimTypeGet(MlirContext wrappedCtx,
                           MlirAttribute wrappedTraitApp) {
  MLIRContext* ctx = unwrap(wrappedCtx);
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type
  return wrap(ClaimType::get(ctx, traitApp));
}

MlirType traitClaimTypeWithApplication(MlirType wrappedClaimType,
                                       MlirAttribute wrappedTraitApp) {
  ClaimType claimType = dyn_cast<ClaimType>(unwrap(wrappedClaimType));
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!claimType || !traitApp) return {};
  return wrap(ClaimType::get(claimType.getContext(), traitApp, claimType.getProof()));
}

MlirAttribute traitClaimTypeGetTraitApplication(MlirType wrappedClaimType) {
  ClaimType claimType = dyn_cast<ClaimType>(unwrap(wrappedClaimType));
  if (!claimType) return {}; // invalid type
  return wrap(claimType.getTraitApplication());
}

bool traitTypeIsAClaim(MlirType type) {
  return isa<ClaimType>(unwrap(type));
}

MlirType traitProjectionTypeGet(MlirContext wrappedCtx,
                                MlirAttribute wrappedTraitApp,
                                MlirStringRef assocName,
                                MlirType *assocTypeArgs, intptr_t numAssocTypeArgs) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {};
  StringAttr nameAttr = StringAttr::get(ctx, StringRef(assocName.data, assocName.length));
  SmallVector<Type> args;
  args.reserve(numAssocTypeArgs);
  for (intptr_t i = 0; i < numAssocTypeArgs; ++i)
    args.push_back(unwrap(assocTypeArgs[i]));
  return wrap(ProjectionType::get(ctx, traitApp, nameAttr, args));
}

bool traitTypeIsAProjection(MlirType type) {
  return isa<ProjectionType>(unwrap(type));
}

bool traitTypeIsGeneric(MlirType type) {
  return isa<GenericTypeInterface>(unwrap(type));
}

bool traitTypeCarriesPolymorphism(MlirType type) {
  return isa<PolymorphicTypeInterface>(unwrap(type));
}

MlirOperation traitProjCastOpCreate(MlirLocation loc,
                                     MlirValue input,
                                     MlirValue claim,
                                     MlirType resultType) {
  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto op = ProjCastOp::create(builder,
    unwrap(loc),
    unwrap(resultType),
    unwrap(input),
    unwrap(claim)
  );

  return wrap(op.getOperation());
}

MlirAttribute traitTypeEqualityAttrGet(MlirContext wrappedCtx,
                                       MlirType lhs, MlirType rhs) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto eq = TypeEqualityAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx, unwrap(lhs),
      unwrap(rhs));
  return wrap(eq);
}

MlirType traitClaimTypeGetEquality(MlirContext wrappedCtx,
                                   MlirType lhs, MlirType rhs) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto eq = TypeEqualityAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx, unwrap(lhs),
      unwrap(rhs));
  if (!eq)
    return {};
  return wrap(ClaimType::getEquality(ctx, eq));
}

bool traitTypeIsAnEqualityClaim(MlirType type) {
  auto claim = dyn_cast<ClaimType>(unwrap(type));
  return claim && claim.isEquality();
}

MlirAttribute traitWitnessCertificateAttrGet(MlirContext wrappedCtx,
                                             MlirType redex, MlirType contractum,
                                             MlirStringRef implName) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  FlatSymbolRefAttr implRef =
      FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length));
  auto cert = WitnessCertificateAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx, unwrap(redex),
      unwrap(contractum), implRef);
  return wrap(cert);
}

MlirOperation traitWitnessProjResolveOpCreate(MlirLocation loc,
                                              MlirAttribute wrappedCert,
                                              MlirValue *premises,
                                              intptr_t numPremises,
                                              MlirType resultType) {
  auto cert = dyn_cast<WitnessCertificateAttr>(unwrap(wrappedCert));
  if (!cert)
    return {};
  auto claim = dyn_cast<ClaimType>(unwrap(resultType));
  if (!claim || !claim.isEquality())
    return {};

  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> premiseVals;
  premiseVals.reserve(numPremises);
  for (intptr_t i = 0; i < numPremises; ++i)
    premiseVals.push_back(unwrap(premises[i]));

  auto op = WitnessOp::create(builder, unwrap(loc), claim.getEqualityAttr(),
                              cert, ValueRange(premiseVals));
  return wrap(op.getOperation());
}

MlirOperation traitWitnessReflOpCreate(MlirLocation loc, MlirType resultType) {
  auto claim = dyn_cast<ClaimType>(unwrap(resultType));
  if (!claim || !claim.isEquality())
    return {};

  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);
  auto op = WitnessOp::create(builder, unwrap(loc), claim.getEqualityAttr());
  return wrap(op.getOperation());
}

MlirOperation traitCoerceOpCreate(MlirLocation loc,
                                  MlirValue input,
                                  MlirValue *equalities, intptr_t numEqualities,
                                  MlirType resultType) {
  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> eqVals;
  eqVals.reserve(numEqualities);
  for (intptr_t i = 0; i < numEqualities; ++i)
    eqVals.push_back(unwrap(equalities[i]));

  auto op = CoerceOp::create(builder, unwrap(loc), unwrap(resultType),
                             unwrap(input), ValueRange(eqVals));
  return wrap(op.getOperation());
}

bool traitWitnessSeamAuditAccepts(MlirModule wrappedModule,
                                  MlirType redex, MlirType contractum,
                                  MlirStringRef implName,
                                  MlirType *premises, intptr_t numPremises,
                                  bool checkObligations) {
  ModuleOp module = unwrap(wrappedModule);
  MLIRContext *ctx = module.getContext();
  FlatSymbolRefAttr implRef =
      FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length));

  // Premises split by arm: the equality claims are the comparison modulus, the
  // application claims discharge the impl's assumptions in obligation mode. In
  // binding mode an application premise is illegal, matching the verifier's
  // equality-only filter.
  SmallVector<TypeEqualityAttr> equalityPremises;
  SmallVector<TraitApplicationAttr> applicationPremises;
  for (intptr_t i = 0; i < numPremises; ++i) {
    auto claim = dyn_cast<ClaimType>(unwrap(premises[i]));
    if (!claim)
      return false;
    if (auto eq = claim.getEqualityAttr()) {
      equalityPremises.push_back(eq);
    } else if (checkObligations && claim.isApplication()) {
      applicationPremises.push_back(claim.getTraitApplication());
    } else {
      return false;
    }
  }

  // A refused audit is a classification answer, not a compile error, so swallow
  // the diagnostics the shared audit emits on refusal.
  ScopedDiagnosticHandler handler(ctx, [](Diagnostic &) { return success(); });
  auto err = [&] { return emitError(UnknownLoc::get(ctx)); };
  return succeeded(auditProjResolveCertificate(
      module, unwrap(redex), unwrap(contractum), implRef, equalityPremises, err,
      applicationPremises, checkObligations));
}

MlirOperation traitAssocTypeOpCreate(MlirLocation loc,
                                     MlirStringRef name,
                                     MlirType boundType,
                                     MlirType *typeParams, intptr_t numTypeParams) {
  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);
  TypeAttr typeAttr = boundType.ptr ? TypeAttr::get(unwrap(boundType))
                                    : TypeAttr();
  ArrayAttr typeParamsAttr;
  if (numTypeParams > 0) {
    SmallVector<Attribute, 4> attrs;
    attrs.reserve(numTypeParams);
    for (intptr_t i = 0; i < numTypeParams; ++i)
      attrs.push_back(TypeAttr::get(unwrap(typeParams[i])));
    typeParamsAttr = ArrayAttr::get(ctx, attrs);
  }
  auto op = AssocTypeOp::create(builder,
    unwrap(loc),
    builder.getStringAttr(StringRef(name.data, name.length)),
    typeAttr,
    typeParamsAttr
  );
  return wrap(op.getOperation());
}

intptr_t traitGetGenericTypesIn(MlirType type, MlirType *results, intptr_t maxResults) {
  auto generics = getGenericTypesIn(unwrap(type));
  intptr_t count = static_cast<intptr_t>(generics.size());
  if (results) {
    intptr_t n = std::min(count, maxResults);
    for (intptr_t i = 0; i < n; ++i)
      results[i] = wrap(generics[i]);
  }
  return count;
}

} // end extern "C"
