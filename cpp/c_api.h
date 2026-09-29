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

/// Return the #trait.predicate_array holding `predicates`, a trait's or an
/// impl's where clause: each entry a trait application, a type equality or a
/// bound predicate. Returns a null attribute if an entry is none of those.
MlirAttribute traitPredicateArrayAttrGet(MlirContext ctx,
                                         MlirAttribute *predicates,
                                         intptr_t numPredicates);

/// Return the !trait.poly<label> type. A label names a position in the
/// declaration that binds it, so it is non-negative.
MlirType traitPolyTypeGet(MlirContext ctx, unsigned int label);

/// Return the unproven !trait.claim over `predicate`: a #trait.application
/// yields an application claim, a #trait.equality an equality claim. Any other
/// attribute yields a null type.
MlirType traitClaimTypeGet(MlirContext ctx,
                           MlirAttribute predicate);

/// Return the !trait.claim<app by @proofName> proving the trait application
/// `traitApp` by the symbol `proofName`. Returns a null type if `traitApp` is no
/// trait application.
MlirType traitProvenClaimTypeGet(MlirAttribute traitApp,
                                 MlirStringRef proofName);

/// Return a claim type with the same proof as `claimType` but
/// with a different trait application.
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
/// #trait.application attribute). Returns a null attribute if `application` is
/// no trait application.
MlirAttribute traitWitnessBodyGetAllegation(MlirContext ctx,
                                            MlirAttribute application);

/// Answer whether `input` and `result` stand under the pending judgment a
/// marked coerce carries, running verifyPendingCoerceEndpoints (TraitOps.hpp).
/// Diagnostics are suppressed; a refusal is a classification answer, not a
/// compile error.
bool traitCoercePendingAccepts(MlirType input, MlirType result);

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

/// The outcome of instantiating an impl a module names.
typedef enum {
  TraitImplInstantiated = 0,
  TraitImplAbsent = 1,
  TraitImplNotItsParameters = 2,
} TraitImplInstantiation;

/// Instantiate the `trait.impl` named `name` at the top level of `module` at
/// `bindings`, #trait type-binding attributes each naming a parameter of the
/// impl and its argument, as a derive stating those arguments does: writes the
/// claim the impl's header states there to `header`, and the claims its
/// where-clause entries state there, in order, up to `maxWhere` of them, into
/// `whereClaims`, with their number in `numWhere`. Writes nothing when `module`
/// holds no impl of that name (`TraitImplAbsent`) or a binding is not a
/// parameter of it and its argument (`TraitImplNotItsParameters`).
TraitImplInstantiation traitModuleInstantiateImpl(MlirModule module, MlirStringRef name,
                                MlirAttribute const *bindings,
                                intptr_t numBindings, MlirType *header,
                                MlirType *whereClaims, intptr_t maxWhere,
                                intptr_t *numWhere);

#ifdef __cplusplus
}
#endif
