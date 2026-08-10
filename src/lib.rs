// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use melior::{
    Context, pass::Pass, StringRef,
    ir::{AttributeLike, Location, Module, Operation, Type, TypeLike, Value, ValueLike},
    ir::attribute::Attribute,
    ir::operation::OperationLike,
};
use mlir_sys::{
    MlirAttribute, MlirContext, MlirLocation, MlirModule,
    MlirOperation, MlirPass, MlirStringRef,
    MlirType, MlirValue,
};

unsafe extern "C" {
    fn traitRegisterDialect(ctx: MlirContext);
    fn traitCreateMonomorphizePass() -> MlirPass;
    fn traitCreateInstantiateMonomorphsPass() -> MlirPass;
    fn traitCreateErasePolymorphsPass() -> MlirPass;

    fn traitTraitApplicationAttrGet(ctx: MlirContext,
                                    trait_name: MlirStringRef,
                                    type_args: *const MlirType, num_type_args: isize) -> MlirAttribute;
    fn traitAttributeIsATraitApplication(attr: MlirAttribute) -> bool;

    fn traitTraitOpCreate(loc: MlirLocation,
                          name: MlirStringRef,
                          type_params: *const MlirType, num_type_params: isize,
                          requirements: *const MlirAttribute, num_requirements: isize) -> MlirOperation;
    fn traitTraitOpCreateWithPredicates(loc: MlirLocation,
                                        name: MlirStringRef,
                                        type_params: *const MlirType, num_type_params: isize,
                                        predicates: *const MlirAttribute, num_predicates: isize) -> MlirOperation;
    fn traitImplOpCreate(loc: MlirLocation,
                         self_trait_app: MlirAttribute,
                         assumptions: *const MlirAttribute, num_assumptions: isize) -> MlirOperation;
    fn traitImplOpCreateNamed(loc: MlirLocation,
                              sym_name: MlirStringRef,
                              self_trait_app: MlirAttribute,
                              assumptions: *const MlirAttribute, num_assumptions: isize) -> MlirOperation;
    fn traitImplOpCreateNamedWithPredicates(loc: MlirLocation,
                                            sym_name: MlirStringRef,
                                            self_trait_app: MlirAttribute,
                                            predicates: *const MlirAttribute, num_predicates: isize) -> MlirOperation;
    fn traitImplOpSetPremises(impl_op: MlirOperation,
                              premises: *const MlirAttribute, num_premises: isize) -> bool;
    fn traitImplOpSetDischarges(impl_op: MlirOperation,
                                discharges: *const MlirAttribute, num_discharges: isize) -> bool;
    fn traitDischargeCitationAttrGet(ctx: MlirContext,
                                     application: MlirAttribute,
                                     impl_name: MlirStringRef) -> MlirAttribute;
    fn traitMethodCallOpCreate(loc: MlirLocation,
                               trait_name: MlirStringRef,
                               method_name: MlirStringRef,
                               claim: MlirValue,
                               arguments: *const MlirValue, num_arguments: isize,
                               result_types: *const MlirType, num_results: isize) -> MlirOperation;
    fn traitFuncCallOpCreate(loc: MlirLocation,
                             callee: MlirStringRef,
                             arguments: *const MlirValue, num_arguments: isize,
                             result_types: *const MlirType, num_results: isize) -> MlirOperation;
    fn traitAllegeOpCreate(loc: MlirLocation,
                           trait_app: MlirAttribute) -> MlirOperation;
    fn traitAllegeUnsafeOpCreate(loc: MlirLocation,
                                 trait_app: MlirAttribute) -> MlirOperation;
    fn traitWitnessOpCreate(loc: MlirLocation,
                            proof_name: MlirStringRef,
                            trait_app: MlirAttribute) -> MlirOperation;
    fn traitProofOpCreate(loc: MlirLocation,
                          sym_name: MlirStringRef,
                          impl_name: MlirStringRef,
                          trait_app: MlirAttribute,
                          subproof_names: *const MlirStringRef, num_subproofs: isize) -> MlirOperation;
    fn traitProjectOpCreate(loc: MlirLocation,
                            src_claim: MlirValue,
                            dest_trait_app: MlirAttribute) -> MlirOperation;
    fn traitProjectOpCreateToClaim(loc: MlirLocation,
                                   src_claim: MlirValue,
                                   result_claim: MlirType) -> MlirOperation;
    fn traitDeriveOpCreate(loc: MlirLocation,
                           trait_app: MlirAttribute,
                           impl_name: MlirStringRef,
                           assumptions: *const MlirValue, num_assumptions: isize) -> MlirOperation;
    fn traitAssumeOpCreate(loc: MlirLocation,
                           trait_app: MlirAttribute) -> MlirOperation;
    fn traitAssumeOpCreateEquality(loc: MlirLocation,
                                   lhs: MlirType, rhs: MlirType) -> MlirOperation;

    fn traitPolyTypeGet(ctx: MlirContext, unique_id: u32) -> MlirType;

    fn traitClaimTypeGet(ctx: MlirContext,
                         trait_app: MlirAttribute) -> MlirType;
    fn traitClaimTypeWithApplication(claim_ty: MlirType,
                                     trait_app: MlirAttribute) -> MlirType;
    fn traitClaimTypeGetTraitApplication(claim_ty: MlirType) -> MlirAttribute;
    fn traitTypeIsAClaim(ty: MlirType) -> bool;
    fn traitGetGenericTypesIn(ty: MlirType, results: *mut MlirType, max_results: isize) -> isize;

    fn traitProjectionTypeGet(ctx: MlirContext,
                              trait_app: MlirAttribute,
                              assoc_name: MlirStringRef,
                              assoc_type_args: *const MlirType, num_assoc_type_args: isize) -> MlirType;
    fn traitTypeIsAProjection(ty: MlirType) -> bool;
    fn traitTypeIsGeneric(ty: MlirType) -> bool;
    fn traitTypeCarriesPolymorphism(ty: MlirType) -> bool;
    fn traitProjCastOpCreate(loc: MlirLocation,
                              input: MlirValue,
                              claim: MlirValue,
                              result_type: MlirType) -> MlirOperation;
    fn traitTypeEqualityAttrGet(ctx: MlirContext,
                                lhs: MlirType, rhs: MlirType) -> MlirAttribute;
    fn traitClaimTypeGetEquality(ctx: MlirContext,
                                 lhs: MlirType, rhs: MlirType) -> MlirType;
    fn traitTypeIsAnEqualityClaim(ty: MlirType) -> bool;
    fn traitWitnessCertificateAttrGet(ctx: MlirContext,
                                      redex: MlirType, contractum: MlirType,
                                      impl_name: MlirStringRef) -> MlirAttribute;
    fn traitWitnessProjResolveOpCreate(loc: MlirLocation,
                                       certificate: MlirAttribute,
                                       premises: *const MlirValue, num_premises: isize,
                                       result_type: MlirType) -> MlirOperation;
    fn traitWitnessReflOpCreate(loc: MlirLocation,
                                result_type: MlirType) -> MlirOperation;
    fn traitWitnessOpCreateCompose(loc: MlirLocation,
                                   result_type: MlirType,
                                   premises: *const MlirValue, num_premises: isize) -> MlirOperation;
    fn traitCoerceOpCreate(loc: MlirLocation,
                           input: MlirValue,
                           equalities: *const MlirValue, num_equalities: isize,
                           result_type: MlirType) -> MlirOperation;
    fn traitWitnessSeamAuditAccepts(module: MlirModule,
                                    redex: MlirType, contractum: MlirType,
                                    impl_name: MlirStringRef,
                                    premises: *const MlirType, num_premises: isize,
                                    check_obligations: bool) -> bool;
    fn traitClaimProjectsTo(module: MlirModule,
                            src_claim: MlirType, dst_claim: MlirType) -> bool;
    fn traitWitnessSeamAuditAcceptsWithDischarges(module: MlirModule,
                                                  redex: MlirType, contractum: MlirType,
                                                  impl_name: MlirStringRef,
                                                  premises: *const MlirType, num_premises: isize,
                                                  discharges: *const MlirAttribute,
                                                  num_discharges: isize) -> bool;
    fn traitAssocTypeOpCreate(loc: MlirLocation,
                              name: MlirStringRef,
                              bound_type: MlirType,
                              type_params: *const MlirType, num_type_params: isize) -> MlirOperation;
}

/// Setting this in the environment makes the monomorphization stage record the
/// demands it declines to serve and write a census of them to standard error
/// when it finishes.
pub const DEMAND_CENSUS_ENVIRONMENT_VARIABLE: &str = "TRAIT_DEMAND_CENSUS";

/// Setting this in the environment arms the ground-projection lookup's per-call
/// check that every monomorphic projection it leaves standing was observed by a
/// recording site, and the stage-exit check that every drainable key still has
/// something left to serve.
pub const DEMAND_CENSUS_CHECK_ENVIRONMENT_VARIABLE: &str = "TRAIT_DEMAND_CENSUS_CHECK";

/// The marker every census line carries.
pub const DEMAND_CENSUS_LINE_PREFIX: &str = "trait-demand-census";

/// The marker a monomorphic projection carries when it survives the lookup with
/// no recording site having observed it. It reports a gap in the ledger's
/// wiring, not a fault in the program being compiled.
pub const DEMAND_CENSUS_UNHOOKED_MARKER: &str = "trait-demand-census unhooked";

/// The marker a drainable key carries when the stage went on to serve it. Like
/// the unhooked marker it reports a gap in the ledger's rules.
pub const DEMAND_CENSUS_SERVED_MARKER: &str = "trait-demand-census served";

/// The marker a type position carries when respelling it by proof-memo lookup
/// and respelling it by a substitution built from the whole memo disagree. Like
/// the two markers above it reports a gap in the dialect's own reasoning.
pub const DEMAND_CENSUS_RESPELLING_DISAGREEMENT_MARKER: &str =
    "trait-demand-census respelling-disagreement";

/// The marker a proof derivation carries when the closure held for it and the
/// closure deriving it again produces differ. Like the markers above it reports
/// a gap in the dialect's own reasoning.
pub const DEMAND_CENSUS_PROOF_DERIVATION_DISAGREEMENT_MARKER: &str =
    "trait-demand-census proof-derivation-disagreement";

/// The marker a deduplicated demanded type carries. These lines name types, and
/// a monomorphic type's name embeds a mangled hash, so they are read where a
/// module is small rather than recorded across a corpus.
pub const DEMAND_CENSUS_DEMAND_MARKER: &str = "trait-demand-census demand";

/// The marker the resolver's scan sizes carry. They move with how much work the
/// resolver does rather than with what the stage demanded.
pub const DEMAND_CENSUS_SCAN_MARKER: &str = "trait-demand-census scan";

/// The marker the population's totals carry. A census that reports nothing
/// still reports this line, so it is what tells a row that raised no demand
/// apart from a row whose census never ran.
pub const DEMAND_CENSUS_SUMMARY_MARKER: &str = "trait-demand-census summary";

/// The marker every line carries that reports what the monomorphization stage
/// did: the facts impl resolution recorded, the rewrite events each run of a
/// greedy pattern driver raised, and how much of the module each
/// claim-respelling sweep touched. These lines share the census switch and
/// carry their own marker.
pub const STAGE_RECORD_LINE_PREFIX: &str = "trait-stage-record";

/// The marker one recorded fact carries. These lines name trait applications,
/// and a monomorphic type's name embeds a mangled hash, so like the census's
/// per-key lines they are read where a module is small rather than recorded
/// across a corpus.
pub const STAGE_RECORD_FACT_MARKER: &str = "trait-stage-record fact";

/// The marker the recorded facts' digest carries, together with the counts
/// behind it.
pub const STAGE_RECORD_DIGEST_MARKER: &str = "trait-stage-record digest";

/// The marker one run of a greedy pattern driver carries.
pub const STAGE_RECORD_REWRITES_MARKER: &str = "trait-stage-record rewrites";

/// The marker one claim-respelling sweep carries.
pub const STAGE_RECORD_RESPELLING_MARKER: &str = "trait-stage-record respelling";

/// The marker one round of the monomorphization stage carries: what it took off
/// the drain, what impl selection settled, and what it rewrote.
pub const STAGE_RECORD_ROUND_MARKER: &str = "trait-stage-record round";

pub fn register(ctx: &Context) {
    unsafe { traitRegisterDialect(ctx.to_raw()) }
}

pub fn create_monomorphize_pass() -> Pass {
    unsafe { Pass::from_raw(traitCreateMonomorphizePass()) }
}

/// The first half of monomorphization: instantiate the monomorphs every trait
/// call needs and prove the monomorphic claims, leaving the polymorphic
/// templates standing.
pub fn create_instantiate_monomorphs_pass() -> Pass {
    unsafe { Pass::from_raw(traitCreateInstantiateMonomorphsPass()) }
}

/// The second half of monomorphization: erase the polymorphic templates, the
/// claims and projections, and the polymorphic function signatures.
pub fn create_erase_polymorphs_pass() -> Pass {
    unsafe { Pass::from_raw(traitCreateErasePolymorphsPass()) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TraitApplicationAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> TraitApplicationAttribute<'c> {
    pub fn new(
        ctx: &'c Context,
        trait_name: &str,
        type_args: &[Type<'c>],
    ) -> Self {
        let attribute = unsafe {
            Attribute::from_raw(traitTraitApplicationAttrGet(
                ctx.to_raw(),
                StringRef::new(trait_name).to_raw(),
                type_args.as_ptr() as *const _,
                type_args.len() as isize,
            ))
        };
        Self { attribute }
    }
}

impl<'c> TryFrom<Attribute<'c>> for TraitApplicationAttribute<'c> {
    type Error = &'static str;

    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        let ok = unsafe { traitAttributeIsATraitApplication(attribute.to_raw()) };
        if ok {
            Ok(Self { attribute })
        } else {
            Err("expected trait::TraitApplicationAttr")
        }
    }
}

impl<'c> From<TraitApplicationAttribute<'c>> for Attribute<'c> {
    fn from(a: TraitApplicationAttribute<'c>) -> Self { a.attribute }
}

impl<'c> AttributeLike<'c> for TraitApplicationAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.attribute.to_raw()
    }
}

impl<'c> std::fmt::Display for TraitApplicationAttribute<'c> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.attribute, f)
    }
}

impl<'c> std::hash::Hash for TraitApplicationAttribute<'c> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.attribute.to_raw().ptr.hash(state);
    }
}

pub fn trait_application_attr<'c>(
    ctx: &'c Context,
    trait_name: &str,
    type_args: &[Type<'c>],
) -> TraitApplicationAttribute<'c> {
    TraitApplicationAttribute::new(
        ctx,
        trait_name,
        type_args,
    )
}

pub fn trait_<'c>(loc: Location<'c>,
                  name: &str,
                  type_params: &[Type<'c>],
                  requirements: &[TraitApplicationAttribute<'c>],
) -> Operation<'c> {
    let req_attrs: Vec<Attribute<'c>> =
        requirements.iter().copied().map(Into::into).collect();
    unsafe { Operation::from_raw(traitTraitOpCreate(
        loc.to_raw(),
        StringRef::new(name).to_raw(),
        type_params.as_ptr() as *const _,
        type_params.len() as isize,
        req_attrs.as_ptr() as *const _,
        req_attrs.len() as isize,
    ))}
}

/// Build a `trait.trait` whose `where` clause carries a mixed list of
/// predicates: each entry is a trait application or a type equality attribute.
pub fn trait_predicates<'c>(loc: Location<'c>,
                            name: &str,
                            type_params: &[Type<'c>],
                            predicates: &[Attribute<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitTraitOpCreateWithPredicates(
        loc.to_raw(),
        StringRef::new(name).to_raw(),
        type_params.as_ptr() as *const _,
        type_params.len() as isize,
        predicates.as_ptr() as *const _,
        predicates.len() as isize,
    ))}
}

pub fn impl_<'c>(loc: Location<'c>,
                 self_trait_app: TraitApplicationAttribute<'c>,
                 assumptions: &[TraitApplicationAttribute<'c>],
) -> Operation<'c> {
    let app_attr: Attribute<'c> = self_trait_app.into();
    let asm_attrs: Vec<Attribute<'c>> =
        assumptions.iter().copied().map(Into::into).collect();
    unsafe { Operation::from_raw(traitImplOpCreate(
        loc.to_raw(),
        app_attr.to_raw(),
        asm_attrs.as_ptr() as *const _,
        asm_attrs.len() as isize,
    ))}
}

pub fn impl_named<'c>(loc: Location<'c>,
                      sym_name: &str,
                      self_trait_app: TraitApplicationAttribute<'c>,
                      assumptions: &[TraitApplicationAttribute<'c>],
) -> Operation<'c> {
    let app_attr: Attribute<'c> = self_trait_app.into();
    let asm_attrs: Vec<Attribute<'c>> =
        assumptions.iter().copied().map(Into::into).collect();
    unsafe { Operation::from_raw(traitImplOpCreateNamed(
        loc.to_raw(),
        StringRef::new(sym_name).to_raw(),
        app_attr.to_raw(),
        asm_attrs.as_ptr() as *const _,
        asm_attrs.len() as isize,
    ))}
}

/// Build a named `trait.impl` whose `where` clause carries a mixed list of
/// predicates: each entry is a trait application the impl assumes, or a type
/// equality it asserts about its own bindings.
pub fn impl_named_with_predicates<'c>(loc: Location<'c>,
                                      sym_name: &str,
                                      self_trait_app: TraitApplicationAttribute<'c>,
                                      predicates: &[Attribute<'c>],
) -> Operation<'c> {
    let app_attr: Attribute<'c> = self_trait_app.into();
    unsafe { Operation::from_raw(traitImplOpCreateNamedWithPredicates(
        loc.to_raw(),
        StringRef::new(sym_name).to_raw(),
        app_attr.to_raw(),
        predicates.as_ptr() as *const _,
        predicates.len() as isize,
    ))}
}

/// Attach projection-resolution premises to an existing `trait.impl` op. Each
/// entry is a `#trait.certificate` attribute resolving a ground sibling
/// projection the impl's own bindings do not; the impl verifier audits them at
/// birth. Returns false if an entry is not a certificate. An empty slice removes
/// any premises the impl already carries. Premises are attached after the impl
/// header prepass completes, so every cited impl is present in the module.
pub fn set_impl_premises<'c>(impl_op: &Operation<'c>, premises: &[Attribute<'c>]) -> bool {
    let raw: Vec<MlirAttribute> = premises.iter().map(|a| a.to_raw()).collect();
    unsafe {
        traitImplOpSetPremises(impl_op.to_raw(), raw.as_ptr(), raw.len() as isize)
    }
}

/// Attach obligation discharge citations to an existing `trait.impl` op. Each
/// entry is a `#trait.discharge` attribute naming an application obligation a
/// cited conditional premise leaves standing and the impl that supplies it; the
/// impl verifier reads them alongside the premises at birth. Returns false if an
/// entry is not a discharge citation. An empty slice removes any citations the
/// impl already carries. Attached after the impl header prepass completes, so
/// every discharger is present in the module.
pub fn set_impl_discharges<'c>(impl_op: &Operation<'c>, discharges: &[Attribute<'c>]) -> bool {
    let raw: Vec<MlirAttribute> = discharges.iter().map(|a| a.to_raw()).collect();
    unsafe {
        traitImplOpSetDischarges(impl_op.to_raw(), raw.as_ptr(), raw.len() as isize)
    }
}

pub fn method_call<'c>(loc: Location<'c>,
                       trait_name: &str,
                       method_name: &str,
                       claim: Value<'c,'_>,
                       arguments: &[Value<'c,'_>],
                       result_types: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitMethodCallOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        StringRef::new(method_name).to_raw(),
        claim.to_raw(),
        arguments.as_ptr() as *const _,
        arguments.len() as isize,
        result_types.as_ptr() as *const _,
        result_types.len() as isize,
    ))}
}

pub fn func_call<'c>(loc: Location<'c>,
                     callee: &str,
                     arguments: &[Value<'c,'_>],
                     result_types: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitFuncCallOpCreate(
        loc.to_raw(),
        StringRef::new(callee).to_raw(),
        arguments.as_ptr() as *const _,
        arguments.len() as isize,
        result_types.as_ptr() as *const _,
        result_types.len() as isize,
    ))}
}

pub fn allege<'c>(loc: Location<'c>,
                  trait_app: TraitApplicationAttribute<'c>,
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitAllegeOpCreate(
        loc.to_raw(),
        trait_app.to_raw(),
    ))}
}

pub fn allege_unsafe<'c>(loc: Location<'c>,
                         trait_app: TraitApplicationAttribute<'c>,
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitAllegeUnsafeOpCreate(
        loc.to_raw(),
        trait_app.to_raw(),
    ))}
}

pub fn witness<'c>(loc: Location<'c>,
                   proof_name: &str,
                   trait_app: TraitApplicationAttribute<'c>,
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitWitnessOpCreate(
        loc.to_raw(),
        StringRef::new(proof_name).to_raw(),
        trait_app.to_raw(),
    ))}
}

pub fn proof<'c>(loc: Location<'c>,
                 sym_name: &str,
                 impl_name: &str,
                 trait_app: TraitApplicationAttribute<'c>,
                 subproof_names: &[&str],
) -> Operation<'c> {
    let raw_names: Vec<MlirStringRef> = subproof_names
        .iter()
        .map(|s| StringRef::new(s).to_raw())
        .collect();
    unsafe { Operation::from_raw(traitProofOpCreate(
        loc.to_raw(),
        StringRef::new(sym_name).to_raw(),
        StringRef::new(impl_name).to_raw(),
        trait_app.to_raw(),
        raw_names.as_ptr(),
        raw_names.len() as isize,
    ))}
}

pub fn project<'c>(loc: Location<'c>,
                   src_claim: Value<'c,'_>,
                   dest_trait_app: TraitApplicationAttribute<'c>,
) -> Operation<'c> {
    let app_attr: Attribute<'c> = dest_trait_app.into();
    unsafe { Operation::from_raw(traitProjectOpCreate(
        loc.to_raw(),
        src_claim.to_raw(),
        app_attr.to_raw(),
    ))}
}

/// Create a `trait.project` op whose result claim is given directly rather than
/// built from a destination trait application. This spells the projection's
/// equality hop: `result_claim` is an equality claim over one of the source
/// trait's requirement endpoints, specialized at the source application.
pub fn project_to_claim<'c>(loc: Location<'c>,
                            src_claim: Value<'c,'_>,
                            result_claim: Type<'c>,
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitProjectOpCreateToClaim(
        loc.to_raw(),
        src_claim.to_raw(),
        result_claim.to_raw(),
    ))}
}

pub fn derive<'c>(loc: Location<'c>,
                  trait_app: TraitApplicationAttribute<'c>,
                  impl_name: &str,
                  assumptions: &[Value<'c,'_>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitDeriveOpCreate(
        loc.to_raw(),
        trait_app.to_raw(),
        StringRef::new(impl_name).to_raw(),
        assumptions.as_ptr() as *const _,
        assumptions.len() as isize,
    ))}
}

pub fn assume<'c>(loc: Location<'c>,
                  trait_app: TraitApplicationAttribute<'c>,
) -> Operation<'c> {
    let app_attr: Attribute<'c> = trait_app.into();
    unsafe { Operation::from_raw(traitAssumeOpCreate(
        loc.to_raw(),
        app_attr.to_raw(),
    ))}
}

/// Build an equality-arm `trait.assume` introducing the hypothesis `lhs = rhs`.
pub fn assume_equality<'c>(loc: Location<'c>,
                          lhs: Type<'c>,
                          rhs: Type<'c>,
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitAssumeOpCreateEquality(
        loc.to_raw(),
        lhs.to_raw(),
        rhs.to_raw(),
    ))}
}

pub fn poly_type<'c>(
    ctx: &'c Context,
    unique_id: u32,
) -> Type<'c> {
    unsafe { Type::from_raw(traitPolyTypeGet(
        ctx.to_raw(),
        unique_id,
    ))}
}

#[derive(Clone, Copy)]
pub struct ClaimType<'c> {
    type_: Type<'c>,
}

impl<'c> ClaimType<'c> {
    pub fn new(ctx: &'c Context,
               trait_app: TraitApplicationAttribute<'c>,
    ) -> Self {
        let type_ = unsafe {
            Type::from_raw(traitClaimTypeGet(
                ctx.to_raw(),
                trait_app.to_raw(),
            ))
        };
        Self { type_ }
    }

    /// Return a claim type with the same proof but a different
    /// trait application.
    pub fn with_application(&self, trait_app: TraitApplicationAttribute<'c>) -> Self {
        let type_ = unsafe {
            Type::from_raw(traitClaimTypeWithApplication(
                self.type_.to_raw(),
                trait_app.to_raw(),
            ))
        };
        Self { type_ }
    }

    pub fn trait_application(&self) -> TraitApplicationAttribute<'c> {
        let attr = unsafe {
            Attribute::from_raw(traitClaimTypeGetTraitApplication(self.type_.to_raw()))
        };
        TraitApplicationAttribute::try_from(attr)
            .expect("C API returned non-TraitApplicationAttr for claim application")
    }
}

impl<'c> TryFrom<Type<'c>> for ClaimType<'c> {
    type Error = &'static str;

    fn try_from(type_: Type<'c>) -> Result<Self, Self::Error> {
        let ok = unsafe { traitTypeIsAClaim(type_.to_raw()) };
        if ok {
            Ok(Self { type_ })
        } else {
            Err("expected trait::ClaimType")
        }
    }
}

impl<'c> TypeLike<'c> for ClaimType<'c> {
    fn to_raw(&self) -> MlirType {
        self.type_.to_raw()
    }
}

impl<'c> From<ClaimType<'c>> for Type<'c> {
    fn from(t: ClaimType<'c>) -> Self { t.type_ }
}

impl<'c> std::fmt::Display for ClaimType<'c> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.type_, f)
    }
}

pub fn claim_type<'c>(
    ctx: &'c Context,
    trait_app: TraitApplicationAttribute<'c>,
) -> ClaimType<'c> {
    ClaimType::new(ctx, trait_app)
}

/// Collect all unique generic types (e.g., !trait.poly, !coord.poly) found
/// recursively in the given type.
pub fn generic_types_in<'c>(ty: Type<'c>) -> Vec<Type<'c>> {
    unsafe {
        let count = traitGetGenericTypesIn(ty.to_raw(), std::ptr::null_mut(), 0);
        let mut results = vec![MlirType { ptr: std::ptr::null_mut() }; count as usize];
        traitGetGenericTypesIn(ty.to_raw(), results.as_mut_ptr(), count);
        results.into_iter().map(|t| Type::from_raw(t)).collect()
    }
}

/// Create a `!trait.proj<@Trait[types], "AssocName", [assoc_type_args]>` type.
pub fn projection_type<'c>(
    ctx: &'c Context,
    trait_app: TraitApplicationAttribute<'c>,
    assoc_name: &str,
    assoc_type_args: &[Type<'c>],
) -> Type<'c> {
    unsafe { Type::from_raw(traitProjectionTypeGet(
        ctx.to_raw(),
        trait_app.to_raw(),
        StringRef::new(assoc_name).to_raw(),
        assoc_type_args.as_ptr() as *const _,
        assoc_type_args.len() as isize,
    ))}
}

/// Check whether a type is a `!trait.claim` type.
pub fn is_claim_type(ty: Type) -> bool {
    unsafe { traitTypeIsAClaim(ty.to_raw()) }
}

/// Check whether a type is a `!trait.proj` type.
pub fn is_projection_type(ty: Type) -> bool {
    unsafe { traitTypeIsAProjection(ty.to_raw()) }
}

/// Check whether a type is a universally-quantified generic — the vocabulary
/// monomorphization substitutes. `!trait.poly`, `!tuple.poly` and
/// `!coord.poly` all answer true.
pub fn is_generic_type(ty: Type) -> bool {
    unsafe { traitTypeIsGeneric(ty.to_raw()) }
}

/// Check whether a type participates in the trait type system's polymorphism.
/// Claim, projection, generic and inference types answer true; a ground type
/// from a dialect outside the trait type system answers false.
pub fn carries_polymorphism(ty: Type) -> bool {
    unsafe { traitTypeCarriesPolymorphism(ty.to_raw()) }
}

/// Create a `trait.proj.cast` op that converts between equivalent types via a claim.
pub fn proj_cast<'c>(
    loc: Location<'c>,
    input: Value<'c, '_>,
    claim: Value<'c, '_>,
    result_type: Type<'c>,
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitProjCastOpCreate(
        loc.to_raw(),
        input.to_raw(),
        claim.to_raw(),
        result_type.to_raw(),
    ))}
}

/// Create an equality-arm `!trait.claim<lhs = rhs>` type. Returns `None` if the
/// endpoints are not receipt-free.
pub fn equality_claim_type<'c>(ctx: &'c Context, lhs: Type<'c>, rhs: Type<'c>) -> Option<Type<'c>> {
    let ty = unsafe { Type::from_raw(traitClaimTypeGetEquality(ctx.to_raw(), lhs.to_raw(), rhs.to_raw())) };
    if ty.to_raw().ptr.is_null() { None } else { Some(ty) }
}

/// The `#trait.equality<lhs = rhs>` predicate attribute. Returns `None` if the
/// endpoints are not receipt-free.
pub fn type_equality_attr<'c>(ctx: &'c Context, lhs: Type<'c>, rhs: Type<'c>) -> Option<Attribute<'c>> {
    let attr = unsafe { Attribute::from_raw(traitTypeEqualityAttrGet(ctx.to_raw(), lhs.to_raw(), rhs.to_raw())) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// Check whether a type is an equality-arm `!trait.claim`.
pub fn is_equality_claim_type(ty: Type) -> bool {
    unsafe { traitTypeIsAnEqualityClaim(ty.to_raw()) }
}

/// The `#trait.certificate<redex resolves contractum by @impl>` attribute frozen
/// into a projection-resolution equality witness. Returns `None` if construction
/// fails.
pub fn witness_certificate_attr<'c>(ctx: &'c Context, redex: Type<'c>, contractum: Type<'c>, impl_name: &str) -> Option<Attribute<'c>> {
    let attr = unsafe { Attribute::from_raw(traitWitnessCertificateAttrGet(
        ctx.to_raw(), redex.to_raw(), contractum.to_raw(), StringRef::new(impl_name).to_raw())) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// The `#trait.discharge<@Application[...] by @impl>` attribute naming `impl_name`
/// as the discharger of the obligation `application` (a `#trait.application`
/// attribute). Returns `None` if `application` is not a trait application.
pub fn discharge_citation_attr<'c>(ctx: &'c Context, application: Attribute<'c>, impl_name: &str) -> Option<Attribute<'c>> {
    let attr = unsafe { Attribute::from_raw(traitDischargeCitationAttrGet(
        ctx.to_raw(), application.to_raw(), StringRef::new(impl_name).to_raw())) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// Create a projection-resolution `trait.witness`. `certificate` is a
/// `#trait.certificate` attribute; `premises` are equality-claim values.
pub fn witness_proj_resolve<'c>(loc: Location<'c>, certificate: Attribute<'c>, premises: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    let raw: Vec<MlirValue> = premises.iter().map(|v| v.to_raw()).collect();
    unsafe { Operation::from_raw(traitWitnessProjResolveOpCreate(
        loc.to_raw(), certificate.to_raw(), raw.as_ptr(), raw.len() as isize, result_type.to_raw())) }
}

/// Create a refl `trait.witness` introducing an `A = A` equality claim.
pub fn witness_refl<'c>(loc: Location<'c>, result_type: Type<'c>) -> Operation<'c> {
    unsafe { Operation::from_raw(traitWitnessReflOpCreate(loc.to_raw(), result_type.to_raw())) }
}

/// Create a composition `trait.witness`. `premises` are equality-claim values
/// whose ground congruence closure entails `result_type` (an equality claim).
/// The witness stores only the leaf premises; the multi-hop equality it names is
/// re-derived at verify by replaying that closure.
pub fn witness_compose<'c>(loc: Location<'c>, premises: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    let raw: Vec<MlirValue> = premises.iter().map(|v| v.to_raw()).collect();
    unsafe { Operation::from_raw(traitWitnessOpCreateCompose(
        loc.to_raw(), result_type.to_raw(), raw.as_ptr(), raw.len() as isize)) }
}

/// Create a `trait.coerce` op: change `input`'s written type to `result_type`,
/// justified by the cited `equalities` (equality-claim values).
pub fn coerce<'c>(loc: Location<'c>, input: Value<'c, '_>, equalities: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    let raw: Vec<MlirValue> = equalities.iter().map(|v| v.to_raw()).collect();
    unsafe { Operation::from_raw(traitCoerceOpCreate(
        loc.to_raw(), input.to_raw(), raw.as_ptr(), raw.len() as isize, result_type.to_raw())) }
}

/// Answer whether the projection-resolution witness seam audit accepts the
/// certificate `(redex, contractum)` cited to `impl_name`, looking that impl up
/// in `module` and matching modulo `premises` (equality-arm claim types, usually
/// empty). This is the same binding-only check `trait.witness`'s equality arm
/// runs at the symbol seam, so a consumer classifying a certificate against this
/// answer cannot disagree with the verifier. Refusal is a plain `false`, not a
/// diagnostic.
pub fn witness_seam_audit_accepts(
    module: &Module,
    redex: Type,
    contractum: Type,
    impl_name: &str,
    premises: &[Type],
) -> bool {
    witness_seam_audit(module, redex, contractum, impl_name, premises, false)
}

/// Whether `src_claim` projects to `dst_claim`: `dst_claim` exactly matches one
/// of the source's candidate projections. This is the exact membership the
/// `trait.project` verifier checks, so codegen can consult it before spelling a
/// projection hop and never disagree with the verifier. Both arguments are claim
/// types; a non-claim argument answers `false`.
pub fn claim_projects_to(module: &Module, src_claim: Type, dst_claim: Type) -> bool {
    unsafe { traitClaimProjectsTo(module.to_raw(), src_claim.to_raw(), dst_claim.to_raw()) }
}

/// The obligation-aware seam audit: like [`witness_seam_audit_accepts`], but the
/// cited impl's own assumptions must additionally be discharged by the
/// application-arm claims among `premises` (the equality-arm claims remain the
/// comparison modulus). This previews the verdict the equality arm's verifier
/// reaches once its audit becomes obligation-aware, so a classifier can refuse
/// exactly what the flipped verifier would before the flip lands.
pub fn witness_seam_audit_accepts_obligation_mode(
    module: &Module,
    redex: Type,
    contractum: Type,
    impl_name: &str,
    premises: &[Type],
) -> bool {
    witness_seam_audit(module, redex, contractum, impl_name, premises, true)
}

/// The obligation-aware seam audit extended with declared discharge citations:
/// like [`witness_seam_audit_accepts_obligation_mode`], but a cited conditional
/// impl's assumption may also be discharged by one of the `discharges` -- each a
/// `#trait.discharge` attribute naming an obligation and the impl that supplies
/// it -- in addition to the application-arm `premises` (the citing impl's own
/// where clause). This previews exactly the verdict the ImplOp verifier reaches
/// with the same premises and discharge citations.
pub fn witness_seam_audit_accepts_obligation_mode_with_discharges<'c>(
    module: &Module,
    redex: Type,
    contractum: Type,
    impl_name: &str,
    premises: &[Type],
    discharges: &[Attribute<'c>],
) -> bool {
    let raw_premises: Vec<MlirType> = premises.iter().map(|t| t.to_raw()).collect();
    let raw_discharges: Vec<MlirAttribute> = discharges.iter().map(|a| a.to_raw()).collect();
    unsafe {
        traitWitnessSeamAuditAcceptsWithDischarges(
            module.to_raw(),
            redex.to_raw(),
            contractum.to_raw(),
            StringRef::new(impl_name).to_raw(),
            raw_premises.as_ptr(),
            raw_premises.len() as isize,
            raw_discharges.as_ptr(),
            raw_discharges.len() as isize,
        )
    }
}

fn witness_seam_audit(
    module: &Module,
    redex: Type,
    contractum: Type,
    impl_name: &str,
    premises: &[Type],
    check_obligations: bool,
) -> bool {
    let raw: Vec<MlirType> = premises.iter().map(|t| t.to_raw()).collect();
    unsafe {
        traitWitnessSeamAuditAccepts(
            module.to_raw(),
            redex.to_raw(),
            contractum.to_raw(),
            StringRef::new(impl_name).to_raw(),
            raw.as_ptr(),
            raw.len() as isize,
            check_obligations,
        )
    }
}

/// Create a `trait.assoc_type` op. Pass `None` for a bare declaration (inside a
/// trait body) or `Some(type)` for a binding (inside an impl body).
/// Pass `type_params` for GAT type parameters (empty slice for non-GAT).
pub fn assoc_type<'c>(loc: Location<'c>, name: &str, bound_type: Option<Type<'c>>, type_params: &[Type<'c>]) -> Operation<'c> {
    let raw_type = match bound_type {
        Some(ty) => ty.to_raw(),
        None => MlirType { ptr: std::ptr::null_mut() },
    };
    unsafe { Operation::from_raw(traitAssocTypeOpCreate(
        loc.to_raw(),
        StringRef::new(name).to_raw(),
        raw_type,
        type_params.as_ptr() as *const _,
        type_params.len() as isize,
    ))}
}
