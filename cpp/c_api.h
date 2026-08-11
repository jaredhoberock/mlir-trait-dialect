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

/// Attach a checked attribute array to a trait.impl operation. When `discharges`
/// is false the array is the impl's projection-resolution premises (each entry a
/// `#trait.certificate` that resolves a ground sibling projection the impl's own
/// bindings do not); when true it is the impl's obligation discharge citations
/// (each entry a `#trait.discharge` naming an application obligation a cited
/// conditional premise leaves standing and the impl that supplies it). An entry
/// of the wrong kind leaves the impl unchanged and returns false. Both arrays are
/// audited by the impl verifier. Attaching an empty array removes the impl's
/// existing entries of that kind.
bool traitImplOpSetCheckedArray(MlirOperation implOp,
                                MlirAttribute* attrs, intptr_t numAttrs,
                                bool discharges);

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

/// Create a trait.project operation whose result claim is given directly. The
/// result is either the destination trait application's claim or a projection's
/// equality hop (an equality claim over a trait requirement's endpoints). Returns
/// a null operation if `resultClaim` is not a claim type.
MlirOperation traitProjectOpCreate(MlirLocation loc,
                                   MlirValue srcClaim,
                                   MlirType resultClaim);

/// Create a trait.derive operation
MlirOperation traitDeriveOpCreate(MlirLocation loc,
                                  MlirAttribute traitApp,
                                  MlirStringRef implName,
                                  MlirValue* assumptions, intptr_t numAssumptions);

/// Create a trait.assume operation introducing the hypothesis `claim`: an
/// application claim `@Trait[...]` or an equality claim `!A = !B`. Yields a null
/// operation if `claim` is not a claim type.
MlirOperation traitAssumeOpCreate(MlirLocation loc, MlirType claim);

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
/// `resultType`. A proven coerce is justified by the cited `equalities`
/// (equality-claim values). A marked coerce (`unproven` true) cites no equalities
/// and stands in the pending judgment its projections discharge at
/// monomorphization.
MlirOperation traitCoerceOpCreate(MlirLocation loc,
                                  MlirValue input,
                                  MlirValue *equalities, intptr_t numEqualities,
                                  MlirType resultType,
                                  bool unproven);

/// Answer whether `input` and `result` could converge under the pending judgment
/// a marked (unproven) trait.coerce carries: the same check `CoerceOp::verify`
/// runs for the marked arm (application-claim receipts stripped, then projection
/// unification with each projection an opaque variable keyed by itself, every
/// other position rigid, bare-projection aliases admitted). Diagnostics are
/// suppressed -- a refusal is a classification answer the frontend consults
/// before routing a site to the marked form, not a compile error.
bool traitCoercePendingAccepts(MlirType input, MlirType result);

/// Answer whether the projection-resolution witness seam audit accepts a
/// certificate. This runs the same obligation-aware audit as trait.witness's
/// equality-arm verifySymbolUses: it looks the impl named by `implName` up in
/// `module`, resolves `redex` through that impl's associated-type binding, applies
/// the equality-claim `premises`, and compares the result against `contractum`;
/// the cited impl's own assumptions must additionally be discharged, either by the
/// application-claim `premises` or by a `discharges` citation (each a
/// `#trait.discharge` attribute naming an obligation and the impl that supplies
/// it). `premises` are !trait.claim types split by arm: equality claims are the
/// comparison modulus, application claims cover the assumptions. When
/// `rigidHeadMatch` is set the redex's application stays rigid, so the verdict
/// never depends on the unrelated impls the module carries -- the impl-birth audit
/// sets it; a witness-site audit leaves it clear. Diagnostics are suppressed -- a
/// refusal is a classification answer, not a compile error.
bool traitWitnessSeamAuditAccepts(MlirModule module,
                                  MlirType redex, MlirType contractum,
                                  MlirStringRef implName,
                                  MlirType *premises, intptr_t numPremises,
                                  MlirAttribute *discharges, intptr_t numDischarges,
                                  bool rigidHeadMatch);

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
