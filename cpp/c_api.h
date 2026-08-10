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

/// Create a trait.trait operation. `requirements` must all be trait
/// applications; use traitTraitOpCreateWithPredicates for a mixed where clause.
MlirOperation traitTraitOpCreate(MlirLocation loc, MlirStringRef name,
                                 MlirType* typeParams, intptr_t numTypeParams,
                                 MlirAttribute* requirements, intptr_t numRequirements);

/// Create a trait.trait operation whose `where` clause carries a mixed list of
/// predicates: each entry is a trait application or a type equality. A
/// non-predicate attribute yields a null operation.
MlirOperation traitTraitOpCreateWithPredicates(MlirLocation loc, MlirStringRef name,
                                               MlirType* typeParams, intptr_t numTypeParams,
                                               MlirAttribute* predicates, intptr_t numPredicates);

/// Create a trait.impl operation. `assumptions` must all be trait applications;
/// use traitImplOpCreateNamedWithPredicates for a mixed where clause.
MlirOperation traitImplOpCreate(MlirLocation loc,
                                MlirAttribute selfTraitApp,
                                MlirAttribute* assumptions, intptr_t numAssumptions);

/// Create a named trait.impl operation. `assumptions` must all be trait
/// applications; use traitImplOpCreateNamedWithPredicates for a mixed clause.
MlirOperation traitImplOpCreateNamed(MlirLocation loc,
                                     MlirStringRef symName,
                                     MlirAttribute selfTraitApp,
                                     MlirAttribute* assumptions, intptr_t numAssumptions);

/// Create a named trait.impl operation whose `where` clause carries a mixed list
/// of predicates: each entry is a trait application the impl assumes, or a type
/// equality it asserts about its own bindings. A non-predicate attribute yields
/// a null operation.
MlirOperation traitImplOpCreateNamedWithPredicates(MlirLocation loc,
                                                   MlirStringRef symName,
                                                   MlirAttribute selfTraitApp,
                                                   MlirAttribute* predicates, intptr_t numPredicates);

/// Attach projection-resolution premises to a trait.impl operation. Each
/// `premises` entry must be a `#trait.certificate` attribute; a non-certificate
/// entry leaves the impl unchanged and returns false. The premises resolve the
/// ground sibling projections the impl's own bindings do not, and are audited by
/// the impl verifier. Attaching an empty array removes any existing premises.
bool traitImplOpSetPremises(MlirOperation implOp,
                            MlirAttribute* premises, intptr_t numPremises);

/// Attach obligation discharge citations to a trait.impl operation. Each
/// `discharges` entry must be a `#trait.discharge` attribute; a non-discharge
/// entry leaves the impl unchanged and returns false. A citation names an
/// application obligation a cited conditional premise leaves standing and the
/// impl that supplies it, so the impl verifier can discharge that assumption
/// without scanning the module. Attaching an empty array removes any existing
/// citations.
bool traitImplOpSetDischarges(MlirOperation implOp,
                              MlirAttribute* discharges, intptr_t numDischarges);

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

/// Create a trait.project operation whose result claim is given directly, rather
/// than built from a destination trait application. This spells the projection's
/// equality hop: the result is an equality claim over a trait requirement's
/// endpoints. Returns a null operation if `resultClaim` is not a claim type.
MlirOperation traitProjectOpCreateToClaim(MlirLocation loc,
                                          MlirValue srcClaim,
                                          MlirType resultClaim);

/// Create a trait.derive operation
MlirOperation traitDeriveOpCreate(MlirLocation loc,
                                  MlirAttribute traitApp,
                                  MlirStringRef implName,
                                  MlirValue* assumptions, intptr_t numAssumptions);

/// Create an application-arm trait.assume operation
MlirOperation traitAssumeOpCreate(MlirLocation loc, MlirAttribute traitApp);

/// Create an equality-arm trait.assume operation introducing `lhs = rhs`.
/// Endpoints must be receipt-free; yields a null operation otherwise.
MlirOperation traitAssumeOpCreateEquality(MlirLocation loc,
                                          MlirType lhs, MlirType rhs);

/// Return the !trait.poly<uniqueId> type
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int uniqueId);

/// Return the !trait.claim<@Trait[Type1, Type2, ...]> type (unproven)
MlirType traitClaimTypeGet(MlirContext ctx,
                           MlirAttribute traitApp);

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

/// Create a trait.proj.cast operation
MlirOperation traitProjCastOpCreate(MlirLocation loc,
                                     MlirValue input,
                                     MlirValue claim,
                                     MlirType resultType);

/// Return the #trait.equality<lhs = rhs> predicate attribute. Endpoints must be
/// receipt-free; returns a null attribute if construction fails.
MlirAttribute traitTypeEqualityAttrGet(MlirContext ctx,
                                       MlirType lhs, MlirType rhs);

/// Return the equality-arm !trait.claim<lhs = rhs> type. Returns a null type if
/// construction fails.
MlirType traitClaimTypeGetEquality(MlirContext ctx,
                                   MlirType lhs, MlirType rhs);

/// Checks whether the given type is an equality-arm claim.
bool traitTypeIsAnEqualityClaim(MlirType type);

/// Return the #trait.certificate<redex resolves contractum by @impl> attribute
/// frozen into a projection-resolution equality witness. Returns a null
/// attribute if construction fails.
MlirAttribute traitWitnessCertificateAttrGet(MlirContext ctx,
                                             MlirType redex, MlirType contractum,
                                             MlirStringRef implName);

/// Return the #trait.discharge<@Application[...] by @impl> attribute that names
/// `implName` as the discharger of the obligation `application` (a
/// `#trait.application` attribute). Returns a null attribute if `application` is
/// not a trait application.
MlirAttribute traitDischargeCitationAttrGet(MlirContext ctx,
                                            MlirAttribute application,
                                            MlirStringRef implName);

/// Create a projection-resolution trait.witness. `certificate` is a
/// #trait.certificate attribute; `premises` are equality-claim values consumed
/// by the projection-headed audit rule. `resultType` is the equality claim.
MlirOperation traitWitnessProjResolveOpCreate(MlirLocation loc,
                                              MlirAttribute certificate,
                                              MlirValue *premises,
                                              intptr_t numPremises,
                                              MlirType resultType);

/// Create a refl trait.witness introducing an A = A equality claim.
MlirOperation traitWitnessReflOpCreate(MlirLocation loc, MlirType resultType);

/// Create a composition trait.witness. `premises` are equality-claim values
/// whose ground congruence closure entails `resultType` (an equality claim).
/// The witness stores only the leaf premises; verify() re-derives the multi-hop
/// equality the result names by replaying that closure.
MlirOperation traitWitnessOpCreateCompose(MlirLocation loc,
                                          MlirType resultType,
                                          MlirValue *premises,
                                          intptr_t numPremises);

/// Create a trait.coerce operation: change `input`'s written type to
/// `resultType`, justified by the cited `equalities` (equality-claim values).
MlirOperation traitCoerceOpCreate(MlirLocation loc,
                                  MlirValue input,
                                  MlirValue *equalities, intptr_t numEqualities,
                                  MlirType resultType);

/// Create a marked (unproven) trait.coerce: change `input`'s written type to
/// `resultType` with no cited equalities, standing in the pending judgment its
/// projections discharge at monomorphization.
MlirOperation traitCoerceOpCreateUnproven(MlirLocation loc,
                                          MlirValue input,
                                          MlirType resultType);

/// Answer whether `input` and `result` could converge under the pending judgment
/// a marked (unproven) trait.coerce carries: the same check `CoerceOp::verify`
/// runs for the marked arm (application-claim receipts stripped, then projection
/// unification with each projection an opaque variable keyed by itself, every
/// other position rigid, bare-projection aliases admitted). Diagnostics are
/// suppressed -- a refusal is a classification answer the frontend consults
/// before routing a site to the marked form, not a compile error.
bool traitCoercePendingAccepts(MlirType input, MlirType result);

/// Answer whether the projection-resolution witness seam audit accepts a
/// certificate. This runs the same check as trait.witness's equality-arm
/// verifySymbolUses: it looks the impl named by `implName` up in `module`,
/// resolves `redex` through that impl's associated-type binding, applies the
/// equality-claim `premises`, and compares the result against `contractum`.
/// `premises` are !trait.claim types.
///
/// When `checkObligations` is false (binding mode, the verifier's verdict) the
/// premises must all be equality claims -- a non-equality premise makes the
/// audit refuse. When true (obligation mode) the premises split by arm: the
/// equality claims are the comparison modulus, and the application claims must
/// discharge the cited impl's own assumptions. Diagnostics are suppressed -- a
/// refusal is a classification answer, not a compile error.
bool traitWitnessSeamAuditAccepts(MlirModule module,
                                  MlirType redex, MlirType contractum,
                                  MlirStringRef implName,
                                  MlirType *premises, intptr_t numPremises,
                                  bool checkObligations);

/// Like `traitWitnessSeamAuditAccepts` in obligation mode, but a cited
/// conditional impl's assumption may also be discharged by one of the
/// `discharges` -- each a `#trait.discharge` attribute naming an obligation and
/// the impl that supplies it -- in addition to the application-arm `premises`
/// (the citing impl's own where clause). This previews exactly the verdict the
/// ImplOp verifier reaches with the same premises and discharge citations, so a
/// front end writing them cannot disagree with the verifier. Diagnostics are
/// suppressed; a refusal is a plain false.
bool traitWitnessSeamAuditAcceptsWithDischarges(MlirModule module,
                                                MlirType redex,
                                                MlirType contractum,
                                                MlirStringRef implName,
                                                MlirType *premises,
                                                intptr_t numPremises,
                                                MlirAttribute *discharges,
                                                intptr_t numDischarges);

/// Whether `srcClaim` projects to `dstClaim`: `dstClaim` exactly matches one of
/// the source's candidate projections (identity, a trait requirement specialized
/// at the source, or a proven impl's assumption). This is the exact membership
/// the ProjectOp verifier checks, exposed so codegen can consult before spelling
/// a projection hop. Returns false if either argument is not a claim type.
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
