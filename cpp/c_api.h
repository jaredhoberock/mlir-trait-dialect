// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Attribute and type getters intern canonical values; boolean queries use the
// dialect's own judgments. These entry points do not mutate existing
// operations. Operations are built through MLIR's generic operation state, and
// each op's verifier is what refuses an ill-formed one.

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the trait dialect with a context.
void traitRegisterDialect(MlirContext ctx);

/// Create an instantiate-monomorphs-trait pass, the first half of
/// monomorphization
MlirPass traitCreateInstantiateMonomorphsPass();

/// Create an erase-polymorphs-trait pass, the second half of monomorphization
MlirPass traitCreateErasePolymorphsPass();

/// Create a TraitApplicationAttr: @Trait[Type...]
MlirAttribute traitTraitApplicationAttrGet(MlirContext ctx,
                                           MlirStringRef traitName,
                                           MlirType *typeArgs, intptr_t numTypeArgs);

/// Checks whether the given attribute is a trait application.
bool traitAttributeIsATraitApplication(MlirAttribute attr);

/// Return the !trait.poly<label> type. A label names a position in the
/// declaration that binds it, so it is non-negative.
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int label);

/// Return the unproven !trait.claim over `predicate`: a #trait.application
/// yields an application claim, a #trait.equality an equality claim. Any other
/// attribute yields a null type.
MlirType traitClaimTypeGet(MlirContext ctx,
                           MlirAttribute predicate);

/// Return the unproven claim of `traitApp`, the claim `claimType` states
/// respelled to another application. A claim names the proof of exactly the
/// application it spells, so the respelled claim names none, whatever proof
/// `claimType` names.
MlirType traitClaimTypeWithApplication(MlirType claimType,
                                       MlirAttribute traitApp);

/// Return a !trait.claim's TraitApplicationAttr
MlirAttribute traitClaimTypeGetTraitApplication(MlirType claimType);

/// Checks whether the given type is a claim type.
bool traitTypeIsAClaim(MlirType type);

/// Return the !trait.proj<@Trait[Types...], "AssocName", [AssocTypeArgs...]> type
MlirType traitProjectionTypeGet(MlirContext ctx,
                                MlirAttribute traitApp,
                                MlirStringRef assocName,
                                MlirType *assocTypeArgs, intptr_t numAssocTypeArgs);

/// Return the #trait.equality<lhs = rhs> predicate attribute. An endpoint must
/// not contain a proven claim; returns a null attribute if construction fails.
MlirAttribute traitTypeEqualityAttrGet(MlirContext ctx,
                                       MlirType lhs, MlirType rhs);

/// Return the #trait.impl_args<[...]> attribute stating `types`, the argument
/// of each of a cited impl's parameters by position.
MlirAttribute traitImplArgumentsAttrGet(MlirContext ctx, const MlirType *types,
                                        intptr_t numTypes);

/// Whether `type` is a !trait.poly.
bool traitTypeIsAPoly(MlirType type);

/// The label of a !trait.poly: the position of the parameter it stands for
/// in the declaration binding it.
unsigned int traitPolyTypeGetLabel(MlirType polyType);

/// The predicate a !trait.claim states: a #trait.application or a
/// #trait.equality.
MlirAttribute traitClaimTypeGetPredicate(MlirType claimType);

/// The symbol name of the trait a #trait.application applies.
MlirStringRef traitTraitApplicationAttrGetTraitName(MlirAttribute traitApp);

/// The number of type arguments of a #trait.application, `Self` first.
intptr_t traitTraitApplicationAttrGetNumTypeArgs(MlirAttribute traitApp);

/// Type argument `index` of a #trait.application.
MlirType traitTraitApplicationAttrGetTypeArg(MlirAttribute traitApp, intptr_t index);

/// Whether the given attribute is a #trait.equality.
bool traitAttributeIsATypeEquality(MlirAttribute attr);

/// The left side of a #trait.equality.
MlirType traitTypeEqualityAttrGetLhs(MlirAttribute equality);

/// The right side of a #trait.equality.
MlirType traitTypeEqualityAttrGetRhs(MlirAttribute equality);

/// Whether `type` is a !trait.proj.
bool traitTypeIsAProjection(MlirType type);

/// The #trait.application a !trait.proj projects out of.
MlirAttribute traitProjectionTypeGetTraitApplication(MlirType projection);

/// The name of the associated type a !trait.proj names.
MlirStringRef traitProjectionTypeGetAssocName(MlirType projection);

/// The number of the associated type's own arguments a !trait.proj states.
intptr_t traitProjectionTypeGetNumAssocTypeArgs(MlirType projection);

/// The associated type's own argument `index` a !trait.proj states.
MlirType traitProjectionTypeGetAssocTypeArg(MlirType projection, intptr_t index);

/// Renames each symbol `names[i]` that an op at the top level of
/// `symbolTableOp` defines to `newNames[i]`, and rewrites every reference whose
/// root names a renamed symbol to name it by its new name, in the attributes,
/// result types and block-argument types of every op inside `symbolTableOp`,
/// nested symbol tables included: a reference is read as the top level's, as
/// this dialect resolves its symbol uses. A reference's nested parts
/// (`@Trait::@method`) are not renamed. An equality claim's endpoints and a
/// citation's arguments, which no replacer moves as sub-elements, move as the
/// clone rule moves them: each rebuilt at once through its constructor.
void traitRenameSymbols(MlirOperation symbolTableOp, const MlirStringRef *names,
                        const MlirStringRef *newNames, intptr_t count);

/// Collect all unique types implementing GenericTypeInterface found in `type`.
///
/// This walks `type` recursively and returns every distinct generic type
/// (e.g., !trait.poly, !coord.poly) encountered. These are the types that
/// would be substituted during monomorphization.
///
/// Call with `results = NULL` to query the count, then call again with a
/// buffer of sufficient size. Returs the total number of unique generic
/// types found.
intptr_t traitGetGenericTypesIn(MlirType type, MlirType *results, intptr_t maxResults);

#ifdef __cplusplus
}
#endif
