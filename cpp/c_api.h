// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Every entry in this API is a pure query: attribute and type getters intern
// canonical values (a null return answers "not well-formed"), the boolean
// entries consult the dialect's own judgments, and nothing here mutates IR.

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

/// Create a trait.trait operation whose `where` clause carries a mixed list of
/// predicates: each entry is a trait application or a type equality. A
/// non-predicate attribute yields a null operation.
MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* typeParams, intptr_t numTypeParams,
                                 MlirAttribute* predicates, intptr_t numPredicates);

/// Create a trait.impl operation. `assumptions` must all be trait applications;
/// use traitImplOpCreateNamed for a named impl with a mixed where clause.
MlirOperation traitImplOpCreate(MlirLocation loc,
                                MlirAttribute selfTraitApp,
                                MlirAttribute* assumptions, intptr_t numAssumptions);

/// Create a named trait.impl operation whose `where` clause carries a mixed list
/// of predicates: each entry is a trait application the impl assumes, or a type
/// equality it asserts about its own bindings. A non-predicate attribute yields
/// a null operation.
MlirOperation traitImplOpCreateNamed(MlirLocation loc,
                                     MlirStringRef symName,
                                     MlirAttribute selfTraitApp,
                                     MlirAttribute* predicates, intptr_t numPredicates);

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

/// Create a trait.derive operation
MlirOperation traitDeriveOpCreate(MlirLocation loc,
                                  MlirAttribute traitApp,
                                  MlirStringRef implName,
                                  MlirValue* assumptions, intptr_t numAssumptions);

/// Return the !trait.poly<uniqueId> type
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int uniqueId);

/// Return the unproven !trait.claim over `predicate`: a #trait.application
/// yields an application claim, a #trait.equality an equality claim. Any other
/// attribute yields a null type.
MlirType traitClaimTypeGet(MlirContext ctx,
                           MlirAttribute predicate);

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

/// Checks whether the given type is a universally-quantified generic, i.e.
/// implements GenericTypeInterface. `!trait.poly`, `!tuple.poly` and
/// `!coord.poly` all answer true; the interface is what monomorphization
/// substitutes.
///
/// XXX TODO: this predicate exists because mlir-c offers no way to ask whether
/// a type implements an interface. It is deleted when a generic
/// interface-implementation query reaches mlir-c upstream.
bool traitTypeIsGeneric(MlirType type);

/// Checks whether the given type participates in the trait type system's
/// polymorphism, i.e. implements PolymorphicTypeInterface. Claim, projection,
/// generic and inference types answer true; a ground type from a dialect
/// outside the trait type system answers false.
///
/// XXX TODO: this predicate exists for the same reason traitTypeIsGeneric
/// does, and is deleted by the same upstream query.
bool traitTypeCarriesPolymorphism(MlirType type);

/// Return the #trait.equality<lhs = rhs> predicate attribute. An endpoint must
/// not contain a proven claim; returns a null attribute if construction fails.
MlirAttribute traitTypeEqualityAttrGet(MlirContext ctx,
                                       MlirType lhs, MlirType rhs);

/// Return the #trait.witness<predicate by @impl> attribute pairing `predicate`
/// with `implName` as the impl that witnesses it. `predicate` is either a type
/// equality (a projection-resolution witness `projection = resolved`) or a
/// `#trait.application` attribute (an obligation the impl discharges). Returns a
/// null attribute if `predicate` is neither arm or construction fails.
MlirAttribute traitWitnessAttrGet(MlirContext ctx,
                                  MlirAttribute predicate,
                                  MlirStringRef implName);

/// Answer whether `input` and `result` converge under the pending judgment a
/// marked coerce carries, running verifyPendingProjectionUnification
/// (TraitOps.hpp). Diagnostics are suppressed; a refusal is a classification
/// answer, not a compile error.
bool traitCoercePendingAccepts(MlirType input, MlirType result);

/// Answer whether a projection-resolution witness cited to `implName` in
/// `module` verifies at a use site, running verifyProjectionResolutionAtUse
/// (TraitOps.hpp). `premises` are !trait.claim types split by arm (equality
/// claims the comparison modulus, application claims covering the cited impl's
/// assumptions); ground projections resolve by module lookup. Diagnostics are
/// suppressed; a refusal is a classification answer, not a compile error.
bool traitProjectionResolutionVerifiesAtUse(MlirModule module,
                                            MlirType projection, MlirType resolved,
                                            MlirStringRef implName,
                                            MlirType *premises, intptr_t numPremises);

/// Answer whether a projection-resolution witness cited to `implName` in
/// `module` verifies at the citing impl's verification, running
/// verifyProjectionResolutionAtImpl (TraitOps.hpp). `premises` are !trait.claim
/// types split by arm (equality claims the comparison modulus, application claims
/// covering the cited impl's assumptions) and `discharges` are `#trait.witness`
/// citations covering the cited impl's conditional assumptions; the projection's
/// application stays rigid. Diagnostics are suppressed; a refusal is a
/// classification answer, not a compile error.
bool traitProjectionResolutionVerifiesAtImpl(MlirModule module,
                                              MlirType projection, MlirType resolved,
                                              MlirStringRef implName,
                                              MlirType *premises, intptr_t numPremises,
                                              MlirAttribute *discharges, intptr_t numDischarges);

/// Whether `srcClaim` projects to `dstClaim`: `dstClaim` exactly matches one of
/// the source's candidate projections (identity, a trait requirement specialized
/// at the source, or a proven impl's assumption). This is the exact membership a
/// projection hop must satisfy. Returns false if either argument is not a claim
/// type.
bool traitClaimProjectsTo(MlirModule module, MlirType srcClaim, MlirType dstClaim);

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
