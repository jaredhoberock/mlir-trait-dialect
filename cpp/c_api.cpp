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

MlirPass traitCreateResolveImplsPass() {
  return wrap(createResolveImplsPass().release());
}

MlirPass traitCreateVerifyAcyclicTraitsPass() {
  return wrap(createVerifyAcyclicTraitsPass().release());
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

  auto op = builder.create<TraitOp>(
    unwrap(loc),
    builder.getStringAttr(StringRef(name.data, name.length)),
    typeParams,
    appAttrs
  );

  return wrap(op.getOperation());
}

MlirOperation traitImplOpCreate(MlirLocation loc, MlirStringRef traitName,
                                MlirType* typeArgs, intptr_t numTypeArgs,
                                MlirAttribute* assumptions, intptr_t numAssumptions) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  SmallVector<Attribute> typeAttrs;
  for (intptr_t i = 0; i < numTypeArgs; ++i) {
    typeAttrs.push_back(TypeAttr::get(unwrap(typeArgs[i])));
  }

  OpBuilder builder(ctx);

  SmallVector<TraitApplicationAttr> appAttrs;
  for (intptr_t i = 0; i < numAssumptions; ++i) {
    auto app = dyn_cast<TraitApplicationAttr>(unwrap(assumptions[i]));
    if (!app) return {}; // invalid type of attribute
    appAttrs.push_back(app);
  }

  // build a TraitApplicationAttr
  auto traitNameAttr = FlatSymbolRefAttr::get(ctx, StringRef(traitName.data, traitName.length));
  auto typeArgsAttr = builder.getArrayAttr(typeAttrs);
  auto traitAppAttr = TraitApplicationAttr::get(ctx, traitNameAttr, typeArgsAttr);

  auto op = builder.create<ImplOp>(
    unwrap(loc),
    traitAppAttr,
    appAttrs
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

  auto op = builder.create<MethodCallOp>(
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

  auto op = builder.create<FuncCallOp>(
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
  auto op = builder.create<AllegeOp>(
    unwrap(loc),
    traitApp
  );

  return wrap(op.getOperation());
}

MlirOperation traitWitnessOpCreate(MlirLocation loc,
                                   MlirStringRef proofName,
                                   MlirStringRef traitName,
                                   MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  MlirAttribute wrappedTraitApp = traitTraitApplicationAttrGet(wrap(ctx), traitName, typeArgs, numTypeArgs);
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  OpBuilder builder(ctx);

  FlatSymbolRefAttr proofRef = FlatSymbolRefAttr::get(ctx, StringRef(proofName.data, proofName.length));

  auto op = builder.create<WitnessOp>(
    unwrap(loc),
    proofRef,
    traitApp
  );

  return wrap(op.getOperation());
}

MlirOperation traitProjectOpCreate(MlirLocation loc,
                                   MlirValue srcClaim,
                                   MlirStringRef traitName,
                                   MlirType* typeArgs, intptr_t numTypeArgs) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  MlirAttribute wrappedDestTraitApp = traitTraitApplicationAttrGet(wrap(ctx), traitName, typeArgs, numTypeArgs);
  MlirType wrappedResultType = traitClaimTypeGet(wrap(ctx), wrappedDestTraitApp);
  ClaimType resultType = dyn_cast<ClaimType>(unwrap(wrappedResultType));
  if (!resultType) return {}; // invalid result type

  OpBuilder builder(ctx);

  auto op = builder.create<ProjectOp>(
    unwrap(loc),
    resultType,
    unwrap(srcClaim)
  );

  return wrap(op.getOperation());
}

MlirOperation traitAssumeOpCreate(MlirLocation loc,
                                  MlirAttribute wrappedTraitApp) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  OpBuilder builder(ctx);

  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {}; // invalid attribute type

  auto op = builder.create<AssumeOp>(unwrap(loc), traitApp);

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

MlirAttribute traitClaimTypeGetTraitApplication(MlirType wrappedClaimType) {
  ClaimType claimType = dyn_cast<ClaimType>(unwrap(wrappedClaimType));
  if (!claimType) return {}; // invalid type
  return wrap(claimType.getTraitApplication());
}

bool traitTypeIsAClaim(MlirType type) {
  return isa<ClaimType>(unwrap(type));
}

} // end extern "C"
