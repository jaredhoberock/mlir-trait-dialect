// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use melior::{
    Context, pass::Pass, StringRef,
    ir::{AttributeLike, Block, Identifier, Location, Operation, Region, RegionLike, Type, TypeLike, Value},
    ir::attribute::Attribute,
    ir::operation::{OperationBuilder, OperationLike},
};
use mlir_sys::{
    MlirAttribute, MlirContext, MlirPass, MlirStringRef, MlirType,
    mlirArrayAttrGet, mlirFlatSymbolRefAttrGet, mlirIdentifierGet, mlirIntegerAttrGet,
    mlirIntegerTypeGet,
    mlirLocationGetContext,
    mlirOperationGetContext,
    mlirOperationSetAttributeByName, mlirStringAttrGet, mlirSymbolRefAttrGet, mlirTypeAttrGet,
    mlirUnitAttrGet,
};

unsafe extern "C" {
    fn traitRegisterDialect(ctx: MlirContext);
    fn traitCreateInstantiateMonomorphsPass() -> MlirPass;
    fn traitCreateErasePolymorphsPass() -> MlirPass;

    fn traitTraitApplicationAttrGet(ctx: MlirContext,
                                    trait_name: MlirStringRef,
                                    type_args: *const MlirType, num_type_args: isize) -> MlirAttribute;
    fn traitAttributeIsATraitApplication(attr: MlirAttribute) -> bool;

    fn traitPredicateArrayAttrGet(ctx: MlirContext,
                                  predicates: *const MlirAttribute, num_predicates: isize) -> MlirAttribute;

    fn traitPolyTypeGet(ctx: MlirContext, label: u32) -> MlirType;

    fn traitClaimTypeGet(ctx: MlirContext,
                         predicate: MlirAttribute) -> MlirType;
    fn traitProvenClaimTypeGet(trait_app: MlirAttribute,
                               proof_name: MlirStringRef) -> MlirType;
    fn traitClaimTypeWithApplication(claim_ty: MlirType,
                                     trait_app: MlirAttribute) -> MlirType;
    fn traitClaimTypeGetTraitApplication(claim_ty: MlirType) -> MlirAttribute;
    fn traitTypeIsAClaim(ty: MlirType) -> bool;
    fn traitGetGenericTypesIn(ty: MlirType, results: *mut MlirType, max_results: isize) -> isize;

    fn traitProjectionTypeGet(ctx: MlirContext,
                              trait_app: MlirAttribute,
                              assoc_name: MlirStringRef,
                              assoc_type_args: *const MlirType, num_assoc_type_args: isize) -> MlirType;
    fn traitTypeEqualityAttrGet(ctx: MlirContext,
                                lhs: MlirType, rhs: MlirType) -> MlirAttribute;
    fn traitTypeBindingAttrGet(ctx: MlirContext,
                               parameter: MlirType, argument: MlirType) -> MlirAttribute;
    fn traitWitnessAttrGet(ctx: MlirContext,
                           predicate: MlirAttribute,
                           impl_name: MlirStringRef,
                           arguments: *const MlirAttribute, num_arguments: isize) -> MlirAttribute;
    fn traitBoundVarTypeGet(ctx: MlirContext, position: u32) -> MlirType;
    fn traitBoundPredicateAttrGet(ctx: MlirContext, arity: u32,
                                  premises: *const MlirAttribute, num_premises: isize,
                                  conclusion: MlirAttribute) -> MlirAttribute;
    fn traitWitnessAttrGetForRequirement(ctx: MlirContext, requirement: u32,
                                         body: MlirAttribute) -> MlirAttribute;
    fn traitWitnessBodyGetCitation(ctx: MlirContext,
                                   impl_name: MlirStringRef,
                                   arguments: *const MlirAttribute, num_arguments: isize,
                                   discharges: *const MlirAttribute, num_discharges: isize) -> MlirAttribute;
    fn traitWitnessBodyGetBinderPremise(ctx: MlirContext, position: u32) -> MlirAttribute;
    fn traitWitnessBodyGetImplPremise(ctx: MlirContext, position: u32) -> MlirAttribute;
    fn traitWitnessBodyGetRequirementHop(ctx: MlirContext, position: u32, of: MlirAttribute,
                                         type_args: *const MlirType, num_type_args: isize,
                                         premises: *const MlirAttribute, num_premises: isize) -> MlirAttribute;
    fn traitWitnessBodyGetAllegation(ctx: MlirContext, application: MlirAttribute) -> MlirAttribute;
    fn traitModuleInstantiateImpl(module: mlir_sys::MlirModule, name: MlirStringRef,
                                  bindings: *const MlirAttribute, num_bindings: isize,
                                  header: *mut MlirType, where_claims: *mut MlirType, max_where: isize,
                                  num_where: *mut isize) -> u32;
}

pub fn register(ctx: &Context) {
    unsafe { traitRegisterDialect(ctx.to_raw()) }
}

/// The first half of monomorphization: instantiate the monomorphs every trait
/// call needs and prove the monomorphic claims, leaving the polymorphic
/// templates standing.
pub fn create_instantiate_monomorphs_pass() -> Pass {
    unsafe { Pass::from_raw(traitCreateInstantiateMonomorphsPass()) }
}

/// The second half of monomorphization: erase the claims, projections and the
/// polymorphic function signatures they stood on, and hold what stands outside a
/// template theory-free. It deletes no template; a `symbol-dce` after it
/// collects the ones nothing names.
pub fn create_erase_polymorphs_pass() -> Pass {
    unsafe { Pass::from_raw(traitCreateErasePolymorphsPass()) }
}

/// Finish a trait op assembled through melior's generic operation builder. These
/// ops declare explicit result types and wire no regions here, so a build
/// failure means a malformed operation state rather than a rejected program; the
/// module verifier run at codegen exit is the authority that refuses an
/// ill-formed op.
fn build_op<'c>(builder: OperationBuilder<'c>) -> Operation<'c> {
    builder.build().expect("trait operation state was malformed")
}

/// A named attribute identifier in the location's context. The context is read
/// as a raw handle rather than a borrowed `&Context`, since the only borrow a
/// `Location` yields is a temporary `ContextRef` that would dangle once bound.
fn identifier<'c>(loc: Location<'c>, name: &str) -> Identifier<'c> {
    unsafe {
        let ctx = mlirLocationGetContext(loc.to_raw());
        Identifier::from_raw(mlirIdentifierGet(ctx, StringRef::new(name).to_raw()))
    }
}

/// A signless 64-bit integer attribute in the location's context, as a
/// positional selector carries (a `trait.project` hop's requirement index).
/// The context is read as a raw handle for the same reason `identifier` reads
/// it that way.
fn index_attr<'c>(loc: Location<'c>, value: usize) -> Attribute<'c> {
    unsafe {
        let ctx = mlirLocationGetContext(loc.to_raw());
        Attribute::from_raw(mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), value as i64))
    }
}

/// An array attribute in the location's context holding `items`.
fn array_attr<'c>(loc: Location<'c>, items: &[MlirAttribute]) -> Attribute<'c> {
    unsafe {
        let ctx = mlirLocationGetContext(loc.to_raw());
        Attribute::from_raw(mlirArrayAttrGet(ctx, items.len() as isize, items.as_ptr()))
    }
}

/// A flat symbol reference to `name` in the location's context.
fn symbol_ref_attr<'c>(loc: Location<'c>, name: &str) -> Attribute<'c> {
    unsafe {
        let ctx = mlirLocationGetContext(loc.to_raw());
        Attribute::from_raw(mlirFlatSymbolRefAttrGet(ctx, StringRef::new(name).to_raw()))
    }
}

/// An array of type attributes in the location's context, as a list of type
/// arguments is stored (a `trait.project` hop's arguments for a binder).
fn type_array_attr<'c>(loc: Location<'c>, types: &[Type<'c>]) -> Attribute<'c> {
    let raw: Vec<MlirAttribute> =
        types.iter().map(|t| unsafe { mlirTypeAttrGet(t.to_raw()) }).collect();
    array_attr(loc, &raw)
}

/// The unit attribute in the location's context (the value of a present
/// `UnitAttr`, e.g. a witness's `refl` or an assume's `self` entry).
fn unit_attr<'c>(loc: Location<'c>) -> Attribute<'c> {
    unsafe { Attribute::from_raw(mlirUnitAttrGet(mlirLocationGetContext(loc.to_raw()))) }
}

/// A string attribute holding `text` in the location's context, as a symbol's
/// name and its visibility are stored.
fn string_attr<'c>(loc: Location<'c>, text: &str) -> Attribute<'c> {
    unsafe {
        Attribute::from_raw(mlirStringAttrGet(
            mlirLocationGetContext(loc.to_raw()), StringRef::new(text).to_raw()))
    }
}

/// The type attribute holding `ty`.
fn type_attr<'c>(ty: Type<'c>) -> Attribute<'c> {
    unsafe { Attribute::from_raw(mlirTypeAttrGet(ty.to_raw())) }
}

/// A declaration's body: one region holding one empty block, which the caller
/// fills through `first_block`.
fn declaration_body<'c>() -> Region<'c> {
    let region = Region::new();
    region.append_block(Block::new(&[]));
    region
}

/// The checked `#trait.predicate_array` holding `predicates`, a where clause.
/// Panics on an entry that is none of a trait application, a type equality and
/// a bound predicate, which is a malformed call rather than a refused program.
fn predicate_array_attr<'c>(loc: Location<'c>, predicates: &[Attribute<'c>]) -> Attribute<'c> {
    let raw: Vec<MlirAttribute> = predicates.iter().map(|p| p.to_raw()).collect();
    let array = unsafe {
        Attribute::from_raw(traitPredicateArrayAttrGet(
            mlirLocationGetContext(loc.to_raw()), raw.as_ptr(), raw.len() as isize))
    };
    assert!(!array.to_raw().ptr.is_null(), "a where clause holds predicates only");
    array
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

/// Build a `trait.trait` whose `where` clause carries a mixed list of
/// predicates: each entry is a trait application or a type equality attribute.
/// A trait is a template that dies with monomorphization, so it is private from
/// birth.
pub fn trait_<'c>(loc: Location<'c>,
                  name: &str,
                  type_params: &[Type<'c>],
                  predicates: &[Attribute<'c>],
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.trait", loc)
        .add_attributes(&[
            (identifier(loc, "sym_name"), string_attr(loc, name)),
            (identifier(loc, "type_params"), type_array_attr(loc, type_params)),
            (identifier(loc, "requirements"), predicate_array_attr(loc, predicates)),
            (identifier(loc, "sym_visibility"), string_attr(loc, "private")),
        ])
        .add_regions([declaration_body()]))
}

/// Build a named `trait.impl` whose `where` clause carries a mixed list of
/// predicates: each entry is a trait application the impl assumes, or a type
/// equality it asserts about its own bindings. An impl is a template that dies
/// with monomorphization, so it is private from birth.
pub fn impl_named<'c>(loc: Location<'c>,
                      sym_name: &str,
                      self_trait_app: TraitApplicationAttribute<'c>,
                      predicates: &[Attribute<'c>],
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.impl", loc)
        .add_attributes(&[
            (identifier(loc, "sym_name"), string_attr(loc, sym_name)),
            (identifier(loc, "self_application"), self_trait_app.into()),
            (identifier(loc, "assumptions"), predicate_array_attr(loc, predicates)),
            (identifier(loc, "sym_visibility"), string_attr(loc, "private")),
        ])
        .add_regions([declaration_body()]))
}

/// Attach the checked `witnesses` array to an existing `trait.impl` op -- each a
/// `#trait.witness` the impl verifier reads by arm: an equality-armed
/// projection-resolution witness, an application-armed obligation discharge
/// covering a cited conditional impl's standing assumption, or the witness of
/// a bound requirement of the impl's trait. The impl
/// verifier checks every entry, its attribute kind included, at impl verification, so this
/// only assembles the array.
pub fn set_impl_witnesses<'c>(
    impl_op: &Operation<'c>,
    attrs: &[Attribute<'c>],
) {
    unsafe {
        let name_ref = StringRef::new("witnesses").to_raw();
        let ctx = mlirOperationGetContext(impl_op.to_raw());
        let raw: Vec<MlirAttribute> = attrs.iter().map(|a| a.to_raw()).collect();
        let array = mlirArrayAttrGet(ctx, raw.len() as isize, raw.as_ptr());
        mlirOperationSetAttributeByName(impl_op.to_raw(), name_ref, array);
    }
}

/// Build a `trait.method.call` of `@trait_name::@method_name` through `claim`,
/// the receiver claim, with `arguments`; `result_types` are the call's results.
pub fn method_call<'c>(loc: Location<'c>,
                       trait_name: &str,
                       method_name: &str,
                       claim: Value<'c,'_>,
                       arguments: &[Value<'c,'_>],
                       result_types: &[Type<'c>],
) -> Operation<'c> {
    let method_ref = unsafe {
        let ctx = mlirLocationGetContext(loc.to_raw());
        let method = mlirFlatSymbolRefAttrGet(ctx, StringRef::new(method_name).to_raw());
        Attribute::from_raw(mlirSymbolRefAttrGet(ctx, StringRef::new(trait_name).to_raw(), 1, &method))
    };
    build_op(OperationBuilder::new("trait.method.call", loc)
        .add_operands(&[claim])
        .add_operands(arguments)
        .add_attributes(&[(identifier(loc, "method_ref"), method_ref)])
        .add_results(result_types))
}

/// Build a `trait.func.call` of `@callee` with `arguments`; `result_types` are
/// the call's results.
pub fn func_call<'c>(loc: Location<'c>,
                     callee: &str,
                     arguments: &[Value<'c,'_>],
                     result_types: &[Type<'c>],
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.func.call", loc)
        .add_operands(arguments)
        .add_attributes(&[(identifier(loc, "callee_name"), symbol_ref_attr(loc, callee))])
        .add_results(result_types))
}

/// The unproven claim of `predicate`, a trait application or a type equality
/// (`type_equality_attr`), in the location's context. Panics on any other
/// attribute, which is a malformed call rather than a refused program.
fn unproven_claim<'c>(loc: Location<'c>, predicate: Attribute<'c>) -> Type<'c> {
    let claim = unsafe {
        Type::from_raw(traitClaimTypeGet(mlirLocationGetContext(loc.to_raw()), predicate.to_raw()))
    };
    assert!(!claim.to_raw().ptr.is_null(), "a claim states a trait application or a type equality");
    claim
}

/// Build a `trait.allege` of the claim predicate `predicate`, a trait
/// application or a type equality (`type_equality_attr`).
pub fn allege<'c>(loc: Location<'c>,
                  predicate: Attribute<'c>,
) -> Operation<'c> {
    let claim = unproven_claim(loc, predicate);
    build_op(OperationBuilder::new("trait.allege", loc).add_results(&[claim]))
}

/// Build a `trait.witness` of `trait_app` proved by the proof or unconditional
/// impl named `proof_name`.
pub fn witness<'c>(loc: Location<'c>,
                   proof_name: &str,
                   trait_app: TraitApplicationAttribute<'c>,
) -> Operation<'c> {
    let claim = unsafe {
        Type::from_raw(traitProvenClaimTypeGet(trait_app.to_raw(), StringRef::new(proof_name).to_raw()))
    };
    build_op(OperationBuilder::new("trait.witness", loc).add_results(&[claim]))
}

/// Create a `trait.project` op selecting the bound requirement `index` of
/// `src_claim` at `type_args`, one per variable it binds, with `premises`, one
/// claim per premise it states there. `result_claim` spells the conclusion that
/// selection derives, which verification checks.
pub fn project_bound<'c>(loc: Location<'c>,
                         src_claim: Value<'c,'_>,
                         index: usize,
                         type_args: &[Type<'c>],
                         premises: &[Value<'c,'_>],
                         result_claim: Type<'c>,
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.project", loc)
        .add_operands(&[src_claim])
        .add_operands(premises)
        .add_attributes(&[(identifier(loc, "index"), index_attr(loc, index)),
                          (identifier(loc, "type_args"), type_array_attr(loc, type_args))])
        .add_results(&[result_claim]))
}

/// Create a `trait.project` op selecting requirement `index` of `src_claim`:
/// its trait's `where` predicates in declaration order, then, when the claim is
/// proven, the assumptions of the impl its proof cites. `result_claim` spells
/// the claim that selection derives, which verification checks.
pub fn project<'c>(loc: Location<'c>,
                   src_claim: Value<'c,'_>,
                   index: usize,
                   result_claim: Type<'c>,
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.project", loc)
        .add_operands(&[src_claim])
        .add_attributes(&[(identifier(loc, "index"), index_attr(loc, index))])
        .add_results(&[result_claim]))
}

/// The `#trait.binding` attributes pairing each of an impl's own parameters, as
/// the impl spells it, with the argument it takes; `None` if a key is not a
/// type parameter.
fn type_bindings<'c>(ctx: MlirContext, arguments: &[(Type<'c>, Type<'c>)]) -> Option<Vec<MlirAttribute>> {
    let mut bindings = Vec::with_capacity(arguments.len());
    for (parameter, argument) in arguments {
        let binding = unsafe { traitTypeBindingAttrGet(ctx, parameter.to_raw(), argument.to_raw()) };
        if binding.ptr.is_null() {
            return None;
        }
        bindings.push(binding);
    }
    Some(bindings)
}

/// Create a `trait.proof` stating the arguments its impl's parameters take,
/// `arguments`, one pair per parameter of the impl. `given` holds one entry per
/// requirement of the trait and per entry of the impl's where clause, in that
/// order: `Some(symbol)` discharging an application entry, `None` for every
/// other. Returns `None` if a key is not a type parameter.
pub fn proof<'c>(loc: Location<'c>,
                 sym_name: &str,
                 impl_name: &str,
                 arguments: &[(Type<'c>, Type<'c>)],
                 trait_app: TraitApplicationAttribute<'c>,
                 given: &[Option<&str>],
) -> Option<Operation<'c>> {
    let bindings = type_bindings(unsafe { mlirLocationGetContext(loc.to_raw()) }, arguments)?;
    let entries: Vec<MlirAttribute> = given
        .iter()
        .map(|entry| match entry {
            Some(symbol) => symbol_ref_attr(loc, symbol).to_raw(),
            None => unit_attr(loc).to_raw(),
        })
        .collect();
    // A proof is a template that dies with monomorphization, so it is private
    // from birth, as every other proof is minted.
    Some(build_op(OperationBuilder::new("trait.proof", loc)
        .add_attributes(&[
            (identifier(loc, "sym_name"), string_attr(loc, sym_name)),
            (identifier(loc, "impl_name"), symbol_ref_attr(loc, impl_name)),
            (identifier(loc, "arguments"), array_attr(loc, &bindings)),
            (identifier(loc, "trait_application"), trait_app.into()),
            (identifier(loc, "subproof_names"), array_attr(loc, &entries)),
            (identifier(loc, "sym_visibility"), string_attr(loc, "private")),
        ])))
}

/// Create a `trait.derive` stating the arguments its impl's parameters take,
/// `arguments`, one pair per parameter of the impl, with `premises` holding one
/// claim per entry of the impl's where clause, in its order. Returns `None` if
/// a key is not a type parameter.
pub fn derive<'c>(loc: Location<'c>,
                  trait_app: TraitApplicationAttribute<'c>,
                  impl_name: &str,
                  arguments: &[(Type<'c>, Type<'c>)],
                  premises: &[Value<'c,'_>],
) -> Option<Operation<'c>> {
    let bindings = type_bindings(unsafe { mlirLocationGetContext(loc.to_raw()) }, arguments)?;
    let claim = unproven_claim(loc, trait_app.into());
    Some(build_op(OperationBuilder::new("trait.derive", loc)
        .add_operands(premises)
        .add_attributes(&[
            (identifier(loc, "impl"), symbol_ref_attr(loc, impl_name)),
            (identifier(loc, "arguments"), array_attr(loc, &bindings)),
        ])
        .add_results(&[claim])))
}

/// Build a positional `trait.assume` citing the self application of the trait
/// or impl whose method it stands in. `claim` spells that application's claim,
/// which verification checks.
pub fn assume_self<'c>(loc: Location<'c>,
                       claim: Type<'c>,
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.assume", loc)
        .add_attributes(&[(identifier(loc, "entry"), unit_attr(loc))])
        .add_results(&[claim]))
}

/// Build a positional `trait.assume` citing entry `position` of the where
/// clause of the trait or impl whose method it stands in. `claim` spells the
/// claim that entry states, which verification checks.
pub fn assume_entry<'c>(loc: Location<'c>,
                        position: usize,
                        claim: Type<'c>,
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.assume", loc)
        .add_attributes(&[(identifier(loc, "entry"), index_attr(loc, position))])
        .add_results(&[claim]))
}

/// The `!trait.poly<label>` type. A label names a position in the declaration
/// that binds it, so it is non-negative and local to that declaration.
pub fn poly_type<'c>(
    ctx: &'c Context,
    label: u32,
) -> Type<'c> {
    unsafe { Type::from_raw(traitPolyTypeGet(
        ctx.to_raw(),
        label,
    ))}
}

/// The `!trait.bound<position>` type: the variable at `position` of the binder
/// of the `#trait.bound` predicate that spells it.
pub fn bound_var_type<'c>(
    ctx: &'c Context,
    position: u32,
) -> Type<'c> {
    unsafe { Type::from_raw(traitBoundVarTypeGet(ctx.to_raw(), position)) }
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


/// Create an equality-arm `!trait.claim<lhs = rhs>` type. Returns `None` if an
/// endpoint contains a proven claim.
pub fn equality_claim_type<'c>(ctx: &'c Context, lhs: Type<'c>, rhs: Type<'c>) -> Option<Type<'c>> {
    let eq = type_equality_attr(ctx, lhs, rhs)?;
    let ty = unsafe { Type::from_raw(traitClaimTypeGet(ctx.to_raw(), eq.to_raw())) };
    if ty.to_raw().ptr.is_null() { None } else { Some(ty) }
}

/// The `#trait.equality<lhs = rhs>` predicate attribute. Returns `None` if an
/// endpoint contains a proven claim.
pub fn type_equality_attr<'c>(ctx: &'c Context, lhs: Type<'c>, rhs: Type<'c>) -> Option<Attribute<'c>> {
    let attr = unsafe { Attribute::from_raw(traitTypeEqualityAttrGet(ctx.to_raw(), lhs.to_raw(), rhs.to_raw())) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// The `#trait.witness<predicate by @impl[!P = T, ...]>` attribute pairing
/// `predicate` (a type equality resolving a projection, or a
/// `#trait.application` the impl discharges) with `impl_name` as the impl that
/// witnesses it. A projection-resolution witness carries the cited impl's
/// substitution: `arguments` pairs each of the impl's own type parameters, as
/// the impl spells it, with the argument it takes. An application witness
/// carries none. Returns `None` if `predicate` is neither arm, a key is not a
/// type parameter, or construction fails.
pub fn witness_attr<'c>(
    ctx: &'c Context,
    predicate: Attribute<'c>,
    impl_name: &str,
    arguments: &[(Type<'c>, Type<'c>)],
) -> Option<Attribute<'c>> {
    let bindings = type_bindings(ctx.to_raw(), arguments)?;
    let attr = unsafe { Attribute::from_raw(traitWitnessAttrGet(
        ctx.to_raw(), predicate.to_raw(), StringRef::new(impl_name).to_raw(),
        bindings.as_ptr(), bindings.len() as isize)) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// The `#trait.bound` predicate `forall [!trait.bound<0>, ...] where
/// [premises] -> conclusion`, a trait's requirement for every choice of
/// `arity` types, spelled with `bound_var_type`. Returns `None` if
/// construction fails: no variable, a premise or conclusion that is neither a
/// trait application nor a type equality or spells a variable past `arity`, or
/// a conclusion spelling no variable.
pub fn bound_predicate_attr<'c>(
    ctx: &'c Context,
    arity: u32,
    premises: &[Attribute<'c>],
    conclusion: Attribute<'c>,
) -> Option<Attribute<'c>> {
    let attr = unsafe { Attribute::from_raw(traitBoundPredicateAttrGet(
        ctx.to_raw(), arity,
        premises.as_ptr() as *const _, premises.len() as isize,
        conclusion.to_raw())) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// A witness body: the evidence a `#trait.witness` states for its predicate.
pub enum WitnessBody<'c> {
    /// The binder's premise at this position.
    BinderPremise(u32),
    /// The stating impl's where-clause entry at this position.
    ImplPremise(u32),
    /// An equality whose sides are one type through the stating impl's bindings.
    Refl,
    /// The impl named, at its parameters' arguments, with one body per entry of
    /// its where clause.
    Citation {
        impl_name: &'c str,
        arguments: Vec<(Type<'c>, Type<'c>)>,
        discharges: Vec<Attribute<'c>>,
    },
    /// Requirement `position` of the application the body `of` proves, at
    /// `type_args`, one per variable the requirement binds, with one body per
    /// premise it states there.
    RequirementHop {
        position: u32,
        of: Attribute<'c>,
        type_args: Vec<Type<'c>>,
        premises: Vec<Attribute<'c>>,
    },
    /// A trait application alleged rather than proved at the stating impl and
    /// proved where the requirement is used.
    Allegation(TraitApplicationAttribute<'c>),
}

/// The witness body attribute `body` describes. Returns `None` if an argument's
/// key is not a type parameter.
pub fn witness_body_attr<'c>(ctx: &'c Context, body: WitnessBody<'c>) -> Option<Attribute<'c>> {
    let raw = unsafe {
        match body {
            WitnessBody::BinderPremise(position) => {
                traitWitnessBodyGetBinderPremise(ctx.to_raw(), position)
            }
            WitnessBody::ImplPremise(position) => {
                traitWitnessBodyGetImplPremise(ctx.to_raw(), position)
            }
            WitnessBody::Refl => mlirUnitAttrGet(ctx.to_raw()),
            WitnessBody::Citation { impl_name, arguments, discharges } => {
                let bindings = type_bindings(ctx.to_raw(), &arguments)?;
                let raw_discharges: Vec<MlirAttribute> =
                    discharges.iter().map(|d| d.to_raw()).collect();
                traitWitnessBodyGetCitation(
                    ctx.to_raw(), StringRef::new(impl_name).to_raw(),
                    bindings.as_ptr(), bindings.len() as isize,
                    raw_discharges.as_ptr(), raw_discharges.len() as isize)
            }
            WitnessBody::RequirementHop { position, of, type_args, premises } => {
                let raw_types: Vec<MlirType> = type_args.iter().map(|t| t.to_raw()).collect();
                let raw_premises: Vec<MlirAttribute> = premises.iter().map(|p| p.to_raw()).collect();
                traitWitnessBodyGetRequirementHop(
                    ctx.to_raw(), position, of.to_raw(),
                    raw_types.as_ptr(), raw_types.len() as isize,
                    raw_premises.as_ptr(), raw_premises.len() as isize)
            }
            WitnessBody::Allegation(application) => {
                traitWitnessBodyGetAllegation(ctx.to_raw(), application.to_raw())
            }
        }
    };
    let attr = unsafe { Attribute::from_raw(raw) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// The `#trait.witness` proving the bound requirement at position `requirement`
/// of the trait of the impl whose `witnesses` array holds it: `body` proves the
/// requirement's conclusion under its binder. Returns `None` if `body` is no
/// witness body.
pub fn requirement_witness_attr<'c>(
    ctx: &'c Context,
    requirement: u32,
    body: Attribute<'c>,
) -> Option<Attribute<'c>> {
    let attr = unsafe { Attribute::from_raw(traitWitnessAttrGetForRequirement(
        ctx.to_raw(), requirement, body.to_raw())) };
    if attr.to_raw().ptr.is_null() { None } else { Some(attr) }
}

/// Create a projection-resolution `trait.witness`. `witness` is an
/// equality-headed `#trait.witness` attribute; `premises` are equality-claim
/// values.
pub fn witness_proj_resolve<'c>(loc: Location<'c>, witness: Attribute<'c>, premises: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.witness", loc)
        .add_attributes(&[(identifier(loc, "witness"), witness)])
        .add_operands(premises)
        .add_results(&[result_type]))
}

/// Create a refl `trait.witness` introducing an `A = A` equality claim.
pub fn witness_refl<'c>(loc: Location<'c>, result_type: Type<'c>) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.witness", loc)
        .add_attributes(&[(identifier(loc, "refl"), unit_attr(loc))])
        .add_results(&[result_type]))
}

/// Create a composition `trait.witness`. `premises` are equality-claim values
/// whose ground congruence closure entails `result_type` (an equality claim).
/// The witness stores only the leaf premises; the multi-hop equality it names is
/// re-derived at verify by replaying that closure.
pub fn witness_compose<'c>(loc: Location<'c>, premises: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.witness", loc)
        .add_operands(premises)
        .add_results(&[result_type]))
}

/// Create a `trait.coerce` op: change `input`'s written type to `result_type`,
/// justified by the cited `equalities` (equality-claim values).
pub fn coerce<'c>(loc: Location<'c>, input: Value<'c, '_>, equalities: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.coerce", loc)
        .add_operands(&[input])
        .add_operands(equalities)
        .add_results(&[result_type]))
}

/// Create a `trait.assoc_type` op. Pass `None` for a bare declaration (inside a
/// trait body) or `Some(type)` for a binding (inside an impl body).
/// Pass `type_params` for GAT type parameters (empty slice for non-GAT).
pub fn assoc_type<'c>(loc: Location<'c>, name: &str, bound_type: Option<Type<'c>>, type_params: &[Type<'c>]) -> Operation<'c> {
    let mut attributes = vec![(identifier(loc, "sym_name"), string_attr(loc, name))];
    if let Some(bound_type) = bound_type {
        attributes.push((identifier(loc, "bound_type"), type_attr(bound_type)));
    }
    if !type_params.is_empty() {
        attributes.push((identifier(loc, "type_params"), type_array_attr(loc, type_params)));
    }
    build_op(OperationBuilder::new("trait.assoc_type", loc).add_attributes(&attributes))
}

/// The outcomes `traitModuleInstantiateImpl` reports, `c_api.h`'s
/// `TraitImplInstantiation`.
const TRAIT_IMPL_INSTANTIATED: u32 = 0;
const TRAIT_IMPL_ABSENT: u32 = 1;

/// Why an impl a module names was not instantiated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImplRefusal {
    /// The module holds no impl of that name.
    Absent,
    /// An argument binds no parameter of the impl.
    NotItsParameters,
}

/// The claims the `trait.impl` named `name` at the top level of `module` states
/// at `arguments`, each a parameter of the impl and the type it takes, as a
/// derive stating those arguments reads them: the claim its header states, and
/// those its where-clause entries state, in order. Refused with `Absent` when
/// the module holds no impl of that name, and with `NotItsParameters` when an
/// argument binds no parameter of it.
pub fn instantiate_impl<'c>(
    ctx: &'c Context,
    module: &melior::ir::Module<'c>,
    name: &str,
    arguments: &[(Type<'c>, Type<'c>)],
) -> Result<(Type<'c>, Vec<Type<'c>>), ImplRefusal> {
    let bindings = type_bindings(ctx.to_raw(), arguments).ok_or(ImplRefusal::NotItsParameters)?;
    let null = MlirType { ptr: std::ptr::null_mut() };
    let mut header = null;
    let mut count = 0isize;
    let instantiate = |header: &mut MlirType, where_claims: &mut [MlirType], count: &mut isize| unsafe {
        traitModuleInstantiateImpl(
            module.to_raw(),
            StringRef::new(name).to_raw(),
            bindings.as_ptr(),
            bindings.len() as isize,
            header,
            where_claims.as_mut_ptr(),
            where_claims.len() as isize,
            count,
        )
    };
    // The first call counts the where clause; the second fills a buffer of
    // that size.
    match instantiate(&mut header, &mut [], &mut count) {
        TRAIT_IMPL_INSTANTIATED => {}
        TRAIT_IMPL_ABSENT => return Err(ImplRefusal::Absent),
        _ => return Err(ImplRefusal::NotItsParameters),
    }
    let mut where_claims = vec![null; count as usize];
    instantiate(&mut header, &mut where_claims, &mut count);
    Ok((
        unsafe { Type::from_raw(header) },
        where_claims.into_iter().map(|claim| unsafe { Type::from_raw(claim) }).collect(),
    ))
}
