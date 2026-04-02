// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the trait dialect with a context.
void traitRegisterDialect(MlirContext ctx);

/// Create a monomorphize-trait pass
MlirPass traitCreateMonomorphizePass();

/// Create a resolve-impls-trait pass
MlirPass traitCreateResolveImplsPass();

/// Create a verify-acyclic-traits pass
MlirPass traitCreateVerifyAcyclicTraitsPass();

/// Create a TraitApplicationAttr: @Trait[Type...]
MlirAttribute traitTraitApplicationAttrGet(MlirContext ctx,
                                           MlirStringRef traitName,
                                           MlirType *typeArgs, intptr_t numTypeArgs);

/// Checks whether the given attribute is a trait application.
bool traitAttributeIsATraitApplication(MlirAttribute attr);

/// Create a trait.trait operation
MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* typeParams, intptr_t numTypeParams,
                                 MlirAttribute* requirements, intptr_t numRequirements);

/// Create a trait.impl operation
MlirOperation traitImplOpCreate(MlirLocation loc,
                                MlirAttribute selfTraitApp,
                                MlirAttribute* assumptions, intptr_t numAssumptions);

/// Create a named trait.impl operation
MlirOperation traitImplOpCreateNamed(MlirLocation loc,
                                     MlirStringRef symName,
                                     MlirAttribute selfTraitApp,
                                     MlirAttribute* assumptions, intptr_t numAssumptions);

/// Create a trait.method.call operation
MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName,
                                      MlirStringRef methodName,
                                      MlirValue claim,
                                      MlirValue* arguments, intptr_t numArguments,
                                      MlirType* resultTypes, intptr_t numResults);

/// Create a trait.func.call operation
MlirOperation traitFuncCallOpCreate(MlirLocation loc,
                                    MlirStringRef callee,
                                    MlirValue* arguments, intptr_t numArguments,
                                    MlirType* resultTypes, intptr_t numResults);

/// Create a trait.allege operation
MlirOperation traitAllegeOpCreate(MlirLocation loc,
                                  MlirAttribute traitApp);

/// Create a trait.allege operation with the unsafe attribute
MlirOperation traitAllegeUnsafeOpCreate(MlirLocation loc,
                                        MlirAttribute traitApp);

/// Create a trait.witness operation
MlirOperation traitWitnessOpCreate(MlirLocation loc,
                                   MlirStringRef proofName,
                                   MlirAttribute traitApp);

/// Create a trait.proof operation
MlirOperation traitProofOpCreate(MlirLocation loc,
                                 MlirStringRef symName,
                                 MlirStringRef implName,
                                 MlirAttribute traitApp,
                                 MlirStringRef* subproofNames, intptr_t numSubproofs);

/// Create a trait.project operation
MlirOperation traitProjectOpCreate(MlirLocation loc,
                                   MlirValue srcClaim,
                                   MlirAttribute destTraitApp);

/// Create a trait.derive operation
MlirOperation traitDeriveOpCreate(MlirLocation loc,
                                  MlirAttribute traitApp,
                                  MlirStringRef implName,
                                  MlirValue* assumptions, intptr_t numAssumptions);

/// Create a trait.assume operation
MlirOperation traitAssumeOpCreate(MlirLocation loc, MlirAttribute traitApp);

/// Return the !trait.poly<uniqueId> type
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int uniqueId);

/// Return the !trait.claim<@Trait[Type1, Type2, ...]> type (unproven)
MlirType traitClaimTypeGet(MlirContext ctx,
                           MlirAttribute traitApp);

/// Return a proven !trait.claim<@Trait[...] by @proof> type.
/// `proofName` is the symbol name of the proof.
MlirType traitClaimTypeGetProven(MlirContext ctx,
                                 MlirAttribute traitApp,
                                 MlirStringRef proofName);

/// Return a claim type with the same proof as `claimType` but
/// with a different trait application.
MlirType traitClaimTypeWithApplication(MlirType claimType,
                                       MlirAttribute traitApp);

/// Return a !trait.claim's TraitApplicationAttr
MlirAttribute traitClaimTypeGetTraitApplicationGet(MlirType claimType);

/// Checks whether the given type is a claim type.
bool traitTypeIsAClaim(MlirType type);

/// Return the !trait.proj<@Trait[Types...], "AssocName", [AssocTypeArgs...]> type
MlirType traitProjectionTypeGet(MlirContext ctx,
                                MlirAttribute traitApp,
                                MlirStringRef assocName,
                                MlirType *assocTypeArgs, intptr_t numAssocTypeArgs);

/// Checks whether the given type is a projection type.
bool traitTypeIsAProjection(MlirType type);

/// Create a trait.proj.cast operation
MlirOperation traitProjCastOpCreate(MlirLocation loc,
                                     MlirValue input,
                                     MlirValue claim,
                                     MlirType resultType);

/// Create a trait.assoc_type op. If boundType.ptr is non-null, the op gets a
/// bound_type attribute (for use inside trait.impl); otherwise it is a bare
/// declaration (for use inside trait.trait).
/// If numTypeParams > 0, typeParams are the GAT type parameters.
MlirOperation traitAssocTypeOpCreate(MlirLocation loc,
                                     MlirStringRef name,
                                     MlirType boundType,
                                     MlirType *typeParams, intptr_t numTypeParams);

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
