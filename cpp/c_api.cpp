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

MlirType traitProvenClaimTypeGet(MlirAttribute wrappedTraitApp,
                                 MlirStringRef proofName) {
  auto traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!traitApp) return {};
  MLIRContext *ctx = traitApp.getContext();
  return wrap(ClaimType::get(
      ctx, traitApp,
      FlatSymbolRefAttr::get(ctx, StringRef(proofName.data, proofName.length))));
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

TraitImplInstantiation traitModuleInstantiateImpl(MlirModule module, MlirStringRef name,
                                MlirType const *parameters,
                                MlirType const *arguments,
                                intptr_t numArguments, MlirType *header,
                                MlirType *whereClaims, intptr_t maxWhere,
                                intptr_t *numWhere) {
  ModuleOp moduleOp = unwrap(module);
  auto impl = dyn_cast_or_null<ImplOp>(
      SymbolTable::lookupSymbolIn(moduleOp, StringRef(name.data, name.length)));
  if (!impl)
    return TraitImplAbsent;
  // The substitution the arguments make, one per parameter of the impl, which
  // the impl's header and where clause are instantiated at.
  SmallVector<GenericTypeInterface, 4> params = impl.getTypeParams();
  SpecializationMap substitution;
  for (intptr_t i = 0; i < numArguments; ++i) {
    auto parameter = dyn_cast<GenericTypeInterface>(unwrap(parameters[i]));
    if (!parameter || !llvm::is_contained(params, parameter) ||
        substitution.lookup(parameter))
      return TraitImplNotItsParameters;
    substitution.bind(parameter, unwrap(arguments[i]));
  }
  if (substitution.bindingCount() != params.size())
    return TraitImplNotItsParameters;
  *header = wrap(Type(
      ClaimType::get(impl.getContext(), impl.getSelfApplicationAt(substitution))));
  SmallVector<ClaimType> where = impl.getWhereClaimsAt(substitution);
  *numWhere = where.size();
  for (auto [position, claim] : llvm::enumerate(where))
    if (intptr_t(position) < maxWhere)
      whereClaims[position] = wrap(Type(claim));
  return TraitImplInstantiated;
}

} // end extern "C"
