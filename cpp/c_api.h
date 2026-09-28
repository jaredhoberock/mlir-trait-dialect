// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Builders return unattached operations. Attribute and type getters intern
// canonical values; boolean queries use the dialect's own judgments. These
// entry points do not mutate existing operations.

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

/// Create a trait.method.call operation. The instance it wants is read off the
/// claim, argument and result types against the method's declaration.
MlirOperation traitMethodCallOpCreate(MlirLocation loc,
                                      MlirStringRef traitName,
                                      MlirStringRef methodName,
                                      MlirValue claim,
                                      MlirValue* arguments, intptr_t numArguments,
                                      MlirType* resultTypes, intptr_t numResults);

/// Create a trait.func.call operation. The instance it wants is read off the
/// operand and result types against the callee's declaration.
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

/// Return the !trait.poly<label> type. A label names a position in the
/// declaration that binds it, so it is non-negative.
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int label);

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

/// Return the #trait.equality<lhs = rhs> predicate attribute. An endpoint must
/// not contain a proven claim; returns a null attribute if construction fails.
MlirAttribute traitTypeEqualityAttrGet(MlirContext ctx,
                                       MlirType lhs, MlirType rhs);

/// Return the #trait.binding<parameter = argument> attribute: one entry of the
/// substitution an impl citation carries, keyed by the impl's own type
/// parameter. Returns a null attribute if `parameter` is not a type parameter.
MlirAttribute traitTypeBindingAttrGet(MlirContext ctx, MlirType parameter,
                                      MlirType argument);

/// Return the #trait.witness<predicate by @impl[arguments]> attribute pairing
/// `predicate` with `implName` as the impl that witnesses it. `predicate` is
/// either a type equality (a projection-resolution witness `projection =
/// resolved`, whose `arguments` are #trait.binding attributes, one per type
/// parameter of the cited impl) or a `#trait.application` attribute (an
/// obligation the impl discharges, carrying no arguments). Returns a null
/// attribute if `predicate` is neither arm or construction fails.
MlirAttribute traitWitnessAttrGet(MlirContext ctx,
                                  MlirAttribute predicate,
                                  MlirStringRef implName,
                                  MlirAttribute *arguments, intptr_t numArguments);

/// Return the !trait.bound<position> type: the variable at `position` of the
/// binder of the #trait.bound predicate that spells it.
MlirType traitBoundVarTypeGet(MlirContext ctx, unsigned int position);

/// Return the #trait.bound predicate `forall [!trait.bound<0>, ...,
/// !trait.bound<arity - 1>] where [premises] -> conclusion`: the requirement a
/// trait states for every choice of `arity` types. Each premise and the
/// conclusion is a trait application or a type equality, and the conclusion
/// spells every variable. Returns a null attribute if construction fails.
MlirAttribute traitBoundPredicateAttrGet(MlirContext ctx,
                                         unsigned int arity,
                                         MlirAttribute *premises,
                                         intptr_t numPremises,
                                         MlirAttribute conclusion);

/// Return the #trait.witness proving the bound requirement at position
/// `requirement` of the trait of the impl whose `witnesses` array holds it,
/// with `body` proving its conclusion under its binder. Returns a null
/// attribute if `body` is no witness body.
MlirAttribute traitWitnessAttrGetForRequirement(MlirContext ctx,
                                                unsigned requirement,
                                                MlirAttribute body);

/// Return the witness body citing the impl `implName` at `arguments`
/// (#trait.binding attributes, one per parameter of that impl), with one body
/// in `discharges` per entry of that impl's where clause, in its order. Returns
/// a null attribute if an argument is not a binding. The body citing the
/// binder's premise and the body citing the impl's own premise are below;
/// reflexivity is the unit attribute.
MlirAttribute traitWitnessBodyGetCitation(MlirContext ctx,
                                          MlirStringRef implName,
                                          MlirAttribute *arguments,
                                          intptr_t numArguments,
                                          MlirAttribute *discharges,
                                          intptr_t numDischarges);

/// Return the witness body citing premise `position` of the binder it stands
/// under.
MlirAttribute traitWitnessBodyGetBinderPremise(MlirContext ctx,
                                               unsigned position);

/// Return the witness body citing entry `position` of the where clause of the
/// impl stating the witness.
MlirAttribute traitWitnessBodyGetImplPremise(MlirContext ctx,
                                             unsigned position);

/// Return the witness body reading requirement `position` of the application
/// the witness body `of` proves, at `typeArgs` (one per variable the
/// requirement binds), with one body in `premises` per premise it states there.
MlirAttribute traitWitnessBodyGetRequirementHop(MlirContext ctx,
                                                unsigned position,
                                                MlirAttribute of,
                                                MlirType *typeArgs,
                                                intptr_t numTypeArgs,
                                                MlirAttribute *premises,
                                                intptr_t numPremises);

/// Return the witness body alleging the trait application `application` (a
/// #trait.application attribute) by the impl rule `rule`, an attribute
/// implementing RuleAttrInterface, or by no rule when `rule` is null. Returns a
/// null attribute if `application` is no trait application or `rule` is
/// neither null nor a rule.
MlirAttribute traitWitnessBodyGetAllegation(MlirContext ctx,
                                            MlirAttribute application,
                                            MlirAttribute rule);

/// Answer whether `input` and `result` stand under the pending judgment a
/// marked coerce carries, running verifyPendingCoerceEndpoints (TraitOps.hpp).
/// Diagnostics are suppressed; a refusal is a classification answer, not a
/// compile error.
bool traitCoercePendingAccepts(MlirType input, MlirType result);

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

/// Run the structural acyclicity screen on `module` and report whether it is
/// free of `where`-clause cycles and dangling trait references. Reads trait
/// symbols by name and refuses a dangling reference cleanly, so it is safe on
/// unverified IR: a launch runs it ahead of the full verifier to screen a frozen
/// blob. Diagnostics reach the context's handler; the return value is the
/// verdict alone. Returns true when the screen holds.
bool traitVerifyAcyclicTraitsStructure(MlirModule module);

/// Whether `op` is a generic trait call instantiation can rewrite now -- a
/// trait.func.call or trait.method.call whose every precondition the instantiate
/// patterns check holds. This is the predicate the instantiate step qualifies
/// its discharge by. Any other op answers false.
bool traitIsRewritableGenericCall(MlirOperation op);

/// Describe the `trait.impl` named `name` at the top level of `module`: the
/// trait it implements; its type parameters, in the order a citation's
/// arguments bind them, up to `maxTypeParams` of them into `typeParams`; and
/// the trait each where-clause entry applies, in order, up to
/// `maxWhereEntries` of them into `whereTraits` (an empty name for an
/// equality entry). The full counts are written to `numTypeParams` and
/// `numWhereEntries`. Returns false, writing nothing, when `module` holds no
/// impl of that name. Names point into the context's storage.
bool traitModuleDescribeImpl(MlirModule module, MlirStringRef name,
                             MlirStringRef *traitName, MlirType *typeParams,
                             intptr_t maxTypeParams, intptr_t *numTypeParams,
                             MlirStringRef *whereTraits,
                             intptr_t maxWhereEntries,
                             intptr_t *numWhereEntries);

/// Whether `module` holds a `trait.trait` named `name` at its top level.
bool traitModuleHasTrait(MlirModule module, MlirStringRef name);

/// Whether `module` still carries instantiation work outside a template: a
/// rewritable generic call, or an unproven monomorphic application claim or an
/// unresolved ground projection not yet discharged. This is the erase step's
/// readiness -- it may run only when this answers false.
bool traitIsPendingExpansion(MlirModule module);

#ifdef __cplusplus
}
#endif
