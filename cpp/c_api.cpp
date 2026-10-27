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
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>

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

MlirType traitClaimTypeWithApplication(MlirType wrappedClaimType,
                                       MlirAttribute wrappedTraitApp) {
  ClaimType claimType = dyn_cast<ClaimType>(unwrap(wrappedClaimType));
  TraitApplicationAttr traitApp = dyn_cast<TraitApplicationAttr>(unwrap(wrappedTraitApp));
  if (!claimType || !traitApp) return {};
  // A claim names the proof of exactly the application it spells
  // (`verifyCitation`), so the claim of another application names none.
  return wrap(ClaimType::get(claimType.getContext(), traitApp));
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

MlirAttribute traitImplArgumentsAttrGet(MlirContext ctx, const MlirType *types,
                                        intptr_t numTypes) {
  SmallVector<Type> unwrapped;
  for (intptr_t i = 0; i < numTypes; ++i)
    unwrapped.push_back(unwrap(types[i]));
  return wrap(ImplArgumentsAttr::get(unwrap(ctx), unwrapped));
}

bool traitTypeIsAPoly(MlirType type) {
  return isa<PolyType>(unwrap(type));
}

unsigned int traitPolyTypeGetLabel(MlirType polyType) {
  return cast<PolyType>(unwrap(polyType)).getLabel();
}

MlirAttribute traitClaimTypeGetPredicate(MlirType claimType) {
  return wrap(cast<ClaimType>(unwrap(claimType)).getPredicate());
}

MlirStringRef traitTraitApplicationAttrGetTraitName(MlirAttribute traitApp) {
  return wrap(cast<TraitApplicationAttr>(unwrap(traitApp)).getTraitName().getValue());
}

intptr_t traitTraitApplicationAttrGetNumTypeArgs(MlirAttribute traitApp) {
  return cast<TraitApplicationAttr>(unwrap(traitApp)).getTypeArgs().size();
}

MlirType traitTraitApplicationAttrGetTypeArg(MlirAttribute traitApp, intptr_t index) {
  return wrap(cast<TraitApplicationAttr>(unwrap(traitApp)).getTypeArgs()[index]);
}

bool traitAttributeIsATypeEquality(MlirAttribute attr) {
  return isa<TypeEqualityAttr>(unwrap(attr));
}

MlirType traitTypeEqualityAttrGetLhs(MlirAttribute equality) {
  return wrap(cast<TypeEqualityAttr>(unwrap(equality)).getLhs());
}

MlirType traitTypeEqualityAttrGetRhs(MlirAttribute equality) {
  return wrap(cast<TypeEqualityAttr>(unwrap(equality)).getRhs());
}

bool traitTypeIsAProjection(MlirType type) {
  return isa<ProjectionType>(unwrap(type));
}

MlirAttribute traitProjectionTypeGetTraitApplication(MlirType projection) {
  return wrap(cast<ProjectionType>(unwrap(projection)).getTraitApplication());
}

MlirStringRef traitProjectionTypeGetAssocName(MlirType projection) {
  return wrap(cast<ProjectionType>(unwrap(projection)).getAssocName().getValue());
}

intptr_t traitProjectionTypeGetNumAssocTypeArgs(MlirType projection) {
  return cast<ProjectionType>(unwrap(projection)).getAssocTypeArgs().size();
}

MlirType traitProjectionTypeGetAssocTypeArg(MlirType projection, intptr_t index) {
  return wrap(cast<ProjectionType>(unwrap(projection)).getAssocTypeArgs()[index]);
}

void traitRenameSymbols(MlirOperation symbolTableOp, const MlirStringRef *names,
                        const MlirStringRef *newNames, intptr_t count) {
  Operation *root = unwrap(symbolTableOp);
  MLIRContext *ctx = root->getContext();
  llvm::DenseMap<StringAttr, StringAttr> renamed;
  for (intptr_t i = 0; i < count; ++i)
    renamed[StringAttr::get(ctx, unwrap(names[i]))] =
        StringAttr::get(ctx, unwrap(newNames[i]));

  // `SymbolTable::replaceAllSymbolUses` rewrites attributes alone and stops at
  // nested symbol tables; a symbol a type names (a claim, a projection, a
  // nominal type) is reached only by replacing elements of types too
  AttrTypeReplacer replacer;
  // A reference's root alone names a top-level symbol; its nested parts name
  // symbols inside it, so the walk does not descend into them.
  replacer.addReplacement(
      [&](SymbolRefAttr reference) -> std::optional<std::pair<Attribute, WalkResult>> {
        auto found = renamed.find(reference.getRootReference());
        if (found == renamed.end())
          return std::make_pair(Attribute(reference), WalkResult::skip());
        return std::make_pair(
            Attribute(SymbolRefAttr::get(found->second, reference.getNestedReferences())),
            WalkResult::skip());
      });
  // An equality claim's endpoints and a citation's arguments are leaves to
  // every replacer; they move as the clone rule moves them, each part rebuilt
  // at once through the attribute's constructor.
  replacer.addReplacement(
      [&](ClaimType claim) -> std::optional<std::pair<Type, WalkResult>> {
        if (claim.isApplication())
          return std::nullopt;
        return respellClaimPredicate(claim, [&](Type t) { return replacer.replace(t); });
      });
  replacer.addReplacement(
      [&](ImplArgumentsAttr arguments) -> std::optional<std::pair<Attribute, WalkResult>> {
        SmallVector<Type> types;
        for (Type type : arguments.getTypes())
          types.push_back(replacer.replace(type));
        return std::make_pair(Attribute(ImplArgumentsAttr::get(ctx, types)), WalkResult::skip());
      });
  replacer.recursivelyReplaceElementsIn(root, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  for (Region &region : root->getRegions())
    for (Block &block : region)
      for (Operation &op : block) {
        auto name = op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
        if (!name)
          continue;
        auto found = renamed.find(name);
        if (found != renamed.end())
          SymbolTable::setSymbolName(&op, found->second);
      }
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
