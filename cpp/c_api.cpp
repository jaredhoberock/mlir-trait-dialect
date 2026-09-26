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
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::trait;

/// Unwrap an array into owned storage for a builder. Empty arrays may be null.
template <typename T>
static auto unwrapArray(T *elements, intptr_t count) {
  SmallVector<decltype(unwrap(*elements))> storage;
  if (count > 0)
    (void)unwrapList(count, elements, storage);
  return storage;
}

/// Check a mixed where-clause before a builder can attach it to an operation.
static PredicateArrayAttr checkedPredicates(MLIRContext *ctx,
                                             MlirAttribute *predicates,
                                             intptr_t count) {
  return PredicateArrayAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx,
      ArrayRef<Attribute>(unwrapArray(predicates, count)));
}

/// Empty parameter arrays are absent, preserving the omitted-argument spelling.
static ArrayAttr typeArrayAttrOrNull(MLIRContext *ctx, MlirType *types,
                                     intptr_t count) {
  return count > 0 ? Builder(ctx).getTypeArrayAttr(unwrapArray(types, count))
                   : ArrayAttr();
}

/// Build an allegation only when its attribute names a trait application.
static MlirOperation createAllegation(MlirLocation loc, MlirAttribute app,
                                      bool isUnsafe) {
  auto traitApp = dyn_cast<TraitApplicationAttr>(unwrap(app));
  if (!traitApp)
    return {};
  OpBuilder builder(unwrap(loc)->getContext());
  return wrap(AllegeOp::create(builder, unwrap(loc), traitApp, isUnsafe)
                  .getOperation());
}

extern "C" {

void traitRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<TraitDialect>();
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

  auto typeArgsAttr = builder.getTypeArrayAttr(unwrapArray(typeArgs, numTypeArgs));

  auto traitRef = FlatSymbolRefAttr::get(
    ctx, StringRef(traitName.data, traitName.length)
  );

  return wrap(TraitApplicationAttr::get(ctx, traitRef, typeArgsAttr));
}

bool traitAttributeIsATraitApplication(MlirAttribute attribute) {
  return isa<TraitApplicationAttr>(unwrap(attribute));
}

MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* wrappedTypeParams, intptr_t numTypeParams,
                                 MlirAttribute* predicates, intptr_t numPredicates) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto typeParams = unwrapArray(wrappedTypeParams, numTypeParams);

  // The mixed where-clause: each predicate is a trait application or a type
  // equality. The array's own verifier judges the arm of every entry; a
  // non-predicate attribute is rejected by returning a null op.
  auto predsAttr = checkedPredicates(ctx, predicates, numPredicates);
  if (!predsAttr)
    return {};

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
                                     MlirAttribute* predicates, intptr_t numPredicates) {
  TraitApplicationAttr selfApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedSelfTraitApp));
  if (!selfApp) return {}; // invalid type of attribute

  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  // The impl's mixed where-clause: application entries are proof obligations,
  // equality entries assert the impl's own bindings. The array's own verifier
  // judges the arm of every entry; a non-predicate attribute is rejected by
  // returning a null op.
  auto predsAttr = checkedPredicates(ctx, predicates, numPredicates);
  if (!predsAttr)
    return {};

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

  auto args = unwrapArray(arguments, numArguments);

  auto results = unwrapArray(resultTypes, numResults);

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

  auto args = unwrapArray(arguments, numArguments);

  auto results = unwrapArray(resultTypes, numResults);

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
  return createAllegation(loc, wrappedTraitApp, /*isUnsafe=*/false);
}

MlirOperation traitAllegeUnsafeOpCreate(MlirLocation loc,
                                        MlirAttribute wrappedTraitApp) {
  return createAllegation(loc, wrappedTraitApp, /*isUnsafe=*/true);
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

MlirOperation traitDeriveOpCreate(MlirLocation loc,
                                  MlirAttribute wrappedTraitApp,
                                  MlirStringRef implName,
                                  MlirValue* assumptions, intptr_t numAssumptions) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  FlatSymbolRefAttr implRef = FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length));

  auto args = unwrapArray(assumptions, numAssumptions);

  OpBuilder builder(ctx);
  auto op = DeriveOp::create(builder,
    unwrap(loc),
    traitApp,
    implRef,
    args
  );

  return wrap(op.getOperation());
}

MlirType traitPolyTypeGet(MlirContext wrappedCtx, unsigned int label) {
  return wrap(PolyType::get(unwrap(wrappedCtx), label));
}

MlirType traitClaimTypeGet(MlirContext wrappedCtx,
                           MlirAttribute wrappedPredicate) {
  MLIRContext* ctx = unwrap(wrappedCtx);
  auto claim = ClaimType::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx,
      unwrap(wrappedPredicate), /*proof=*/FlatSymbolRefAttr());
  return wrap(claim);
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
  auto args = unwrapArray(assocTypeArgs, numAssocTypeArgs);
  return wrap(ProjectionType::get(ctx, traitApp, nameAttr, args));
}

MlirAttribute traitTypeEqualityAttrGet(MlirContext wrappedCtx,
                                       MlirType lhs, MlirType rhs) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto eq = TypeEqualityAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx, unwrap(lhs),
      unwrap(rhs));
  return wrap(eq);
}

MlirAttribute traitTypeBindingAttrGet(MlirContext wrappedCtx,
                                      MlirType parameter, MlirType argument) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto err = [&] { return emitError(UnknownLoc::get(ctx)); };
  return wrap(TypeBindingAttr::getChecked(err, ctx, unwrap(parameter),
                                          unwrap(argument)));
}

MlirAttribute traitWitnessAttrGet(MlirContext wrappedCtx,
                                  MlirAttribute predicate,
                                  MlirStringRef implName,
                                  MlirAttribute *arguments, intptr_t numArguments) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto err = [&] { return emitError(UnknownLoc::get(ctx)); };
  FlatSymbolRefAttr implRef =
      FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length));
  SmallVector<TypeBindingAttr> bindings;
  for (Attribute argument : unwrapArray(arguments, numArguments)) {
    auto binding = dyn_cast<TypeBindingAttr>(argument);
    if (!binding)
      return {};
    bindings.push_back(binding);
  }
  auto witness = WitnessAttr::getChecked(err, ctx, unwrap(predicate), implRef,
                                         ArrayRef<TypeBindingAttr>(bindings));
  return wrap(witness);
}

bool traitCoercePendingAccepts(MlirType input, MlirType result) {
  // The consult runs the verifier's own marked arm: strip proofs, then the
  // shared pending judgment. Sharing the function keeps the classifier's verdict
  // and the codegen-exit verifier's from ever disagreeing.
  Type in = stripClaimProofs(unwrap(input));
  Type out = stripClaimProofs(unwrap(result));
  // A refused pending judgment is a classification answer, not a compile error,
  // so this consult passes no diagnostic sink and the judgment stays silent.
  return succeeded(verifyPendingCoerceEndpoints(in, out));
}

MlirOperation traitAssocTypeOpCreate(MlirLocation loc,
                                     MlirStringRef name,
                                     MlirType boundType,
                                     MlirType *typeParams, intptr_t numTypeParams) {
  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);
  TypeAttr typeAttr = boundType.ptr ? TypeAttr::get(unwrap(boundType))
                                    : TypeAttr();
  auto typeParamsAttr = typeArrayAttrOrNull(ctx, typeParams, numTypeParams);
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

bool traitVerifyAcyclicTraitsStructure(MlirModule module) {
  return succeeded(verifyAcyclicTraitsStructure(unwrap(module)));
}

bool traitIsRewritableGenericCall(MlirOperation op) {
  return isRewritableGenericCall(unwrap(op));
}

bool traitIsPendingExpansion(MlirModule module) {
  return isPendingExpansion(unwrap(module));
}

} // end extern "C"
