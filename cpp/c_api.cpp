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

MlirAttribute traitPredicateArrayAttrGet(MlirContext wrappedCtx,
                                         MlirAttribute *predicates,
                                         intptr_t numPredicates) {
  // The array's own verifier judges the arm of every entry, so a non-predicate
  // attribute yields a null attribute rather than an ill-formed where clause.
  MLIRContext *ctx = unwrap(wrappedCtx);
  return wrap(PredicateArrayAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(ctx)); }, ctx,
      ArrayRef<Attribute>(unwrapArray(predicates, numPredicates))));
}

/// The #trait.binding attributes `arguments` holds, or nothing when one is of
/// another kind.
static std::optional<SmallVector<TypeBindingAttr>>
unwrapBindings(MlirAttribute *arguments, intptr_t count) {
  SmallVector<TypeBindingAttr> bindings;
  for (Attribute argument : unwrapArray(arguments, count)) {
    auto binding = dyn_cast<TypeBindingAttr>(argument);
    if (!binding)
      return std::nullopt;
    bindings.push_back(binding);
  }
  return bindings;
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

bool traitClaimTypeIsMonomorphic(MlirType claimType) {
  return cast<ClaimType>(unwrap(claimType)).isMonomorphic();
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
  auto bindings = unwrapBindings(arguments, numArguments);
  if (!bindings)
    return {};
  return wrap(WitnessAttr::getChecked(
      err, ctx, unwrap(predicate),
      FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length)),
      ArrayRef<TypeBindingAttr>(*bindings)));
}

MlirAttribute traitWitnessAttrGetForRequirement(MlirContext wrappedCtx,
                                                unsigned requirement,
                                                MlirAttribute body) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto err = [&] { return emitError(UnknownLoc::get(ctx)); };
  Attribute position = IntegerAttr::get(IntegerType::get(ctx, 64), requirement);
  return wrap(WitnessAttr::getChecked(err, ctx, position, unwrap(body)));
}

MlirAttribute traitWitnessBodyGetCitation(MlirContext wrappedCtx,
                                          MlirStringRef implName,
                                          MlirAttribute *arguments,
                                          intptr_t numArguments,
                                          MlirAttribute *discharges,
                                          intptr_t numDischarges) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto bindings = unwrapBindings(arguments, numArguments);
  if (!bindings)
    return {};
  return wrap(ImplCitationAttr::get(
      ctx, FlatSymbolRefAttr::get(ctx, StringRef(implName.data, implName.length)),
      *bindings, unwrapArray(discharges, numDischarges)));
}

MlirAttribute traitWitnessBodyGetBinderPremise(MlirContext ctx,
                                               unsigned position) {
  return wrap(BinderPremiseAttr::get(unwrap(ctx), position));
}

MlirAttribute traitWitnessBodyGetImplPremise(MlirContext ctx,
                                             unsigned position) {
  return wrap(ImplPremiseAttr::get(unwrap(ctx), position));
}

MlirAttribute traitWitnessBodyGetRequirementHop(MlirContext ctx,
                                                unsigned position,
                                                MlirAttribute of,
                                                MlirType *typeArgs,
                                                intptr_t numTypeArgs,
                                                MlirAttribute *premises,
                                                intptr_t numPremises) {
  SmallVector<Type> types;
  for (intptr_t i = 0; i < numTypeArgs; ++i)
    types.push_back(unwrap(typeArgs[i]));
  return wrap(RequirementHopAttr::get(unwrap(ctx), position, unwrap(of), types,
                                      unwrapArray(premises, numPremises)));
}

MlirAttribute traitWitnessBodyGetAllegation(MlirContext ctx,
                                            MlirAttribute application) {
  auto app = dyn_cast_or_null<TraitApplicationAttr>(unwrap(application));
  if (!app)
    return {};
  return wrap(AllegationAttr::get(unwrap(ctx), app));
}

MlirType traitBoundVarTypeGet(MlirContext wrappedCtx, unsigned int position) {
  return wrap(BoundVarType::get(unwrap(wrappedCtx), position));
}

MlirAttribute traitBoundPredicateAttrGet(MlirContext wrappedCtx,
                                         unsigned int arity,
                                         MlirAttribute *premises,
                                         intptr_t numPremises,
                                         MlirAttribute conclusion) {
  MLIRContext *ctx = unwrap(wrappedCtx);
  auto err = [&] { return emitError(UnknownLoc::get(ctx)); };
  return wrap(BoundPredicateAttr::getChecked(
      err, ctx, arity, ArrayRef<Attribute>(unwrapArray(premises, numPremises)),
      unwrap(conclusion)));
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
                                MlirAttribute const *bindings,
                                intptr_t numBindings, MlirType *header,
                                MlirType *whereClaims, intptr_t maxWhere,
                                intptr_t *numWhere) {
  ModuleOp moduleOp = unwrap(module);
  auto impl = dyn_cast_or_null<ImplOp>(
      SymbolTable::lookupSymbolIn(moduleOp, StringRef(name.data, name.length)));
  if (!impl)
    return TraitImplAbsent;
  SmallVector<TypeBindingAttr> arguments;
  for (intptr_t i = 0; i < numBindings; ++i) {
    auto binding = dyn_cast_or_null<TypeBindingAttr>(unwrap(bindings[i]));
    if (!binding)
      return TraitImplNotItsParameters;
    arguments.push_back(binding);
  }
  // The substitution a derive stating these arguments makes, and the header
  // and where clause its verifier compares at it.
  FailureOr<SpecializationMap> substitution =
      impl.substitutionFor(arguments, /*err=*/nullptr);
  if (failed(substitution))
    return TraitImplNotItsParameters;
  *header = wrap(Type(
      ClaimType::get(impl.getContext(), impl.getSelfApplicationAt(*substitution))));
  SmallVector<ClaimType> where = impl.getWhereClauseAt(*substitution);
  *numWhere = where.size();
  for (auto [position, claim] : llvm::enumerate(where))
    if (intptr_t(position) < maxWhere)
      whereClaims[position] = wrap(Type(claim));
  return TraitImplInstantiated;
}

} // end extern "C"
