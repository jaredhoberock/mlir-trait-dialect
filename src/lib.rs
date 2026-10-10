// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use melior::{
    Context, pass::Pass, StringRef,
    ir::{AttributeLike, Block, Identifier, Location, Operation, Region, RegionLike, Type, TypeLike, Value},
    ir::attribute::Attribute,
    ir::operation::OperationBuilder,
};
use mlir_sys::{
    MlirAttribute, MlirContext, MlirPass, MlirStringRef, MlirType,
    mlirArrayAttrGet, mlirFlatSymbolRefAttrGet, mlirIdentifierGet, mlirIntegerAttrGet,
    mlirIntegerTypeGet,
    mlirLocationGetContext,
    mlirStringAttrGet, mlirSymbolRefAttrGet, mlirTypeAttrGet,
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

    fn traitPolyTypeGet(ctx: MlirContext, label: u32) -> MlirType;

    fn traitClaimTypeGet(ctx: MlirContext,
                         predicate: MlirAttribute) -> MlirType;
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

/// An array of type attributes in the location's context, as a list of types
/// is stored (a trait's requirements).
fn type_array_attr<'c>(loc: Location<'c>, types: &[Type<'c>]) -> Attribute<'c> {
    let raw: Vec<MlirAttribute> =
        types.iter().map(|t| unsafe { mlirTypeAttrGet(t.to_raw()) }).collect();
    array_attr(loc, &raw)
}

/// The unit attribute in the location's context (the value of a present
/// `UnitAttr`, e.g. a witness's `refl`).
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

/// A declaration's body: one region holding one block taking `arguments`,
/// which the caller fills through `first_block`.
fn declaration_body<'c>(loc: Location<'c>, arguments: &[Type<'c>]) -> Region<'c> {
    let region = Region::new();
    let arguments: Vec<(Type<'c>, Location<'c>)> = arguments.iter().map(|ty| (*ty, loc)).collect();
    region.append_block(Block::new(&arguments));
    region
}

/// Build a `trait.trait` whose one block argument is `self_claim`, the claim of
/// the trait's own application, and whose result signature is `requirements`,
/// one claim per requirement in order. A trait is a template that dies with
/// monomorphization, so it is private from birth.
pub fn trait_<'c>(loc: Location<'c>,
                  name: &str,
                  self_claim: Type<'c>,
                  requirements: &[Type<'c>],
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.trait", loc)
        .add_attributes(&[
            (identifier(loc, "sym_name"), string_attr(loc, name)),
            (identifier(loc, "requirements"), type_array_attr(loc, requirements)),
            (identifier(loc, "sym_visibility"), string_attr(loc, "private")),
        ])
        .add_regions([declaration_body(loc, &[self_claim])]))
}

/// Build a named `trait.impl` whose block arguments are `self_claim`, the
/// claim of the application it implements, and then `where_claims`, one claim
/// per where entry in order. The caller fills the body and ends it with
/// `trait.return` of the evidence for the trait's requirements. An impl is a
/// template that dies with monomorphization, so it is private from birth.
pub fn impl_named<'c>(loc: Location<'c>,
                      sym_name: &str,
                      self_claim: Type<'c>,
                      where_claims: &[Type<'c>],
) -> Operation<'c> {
    let mut arguments = vec![self_claim];
    arguments.extend_from_slice(where_claims);
    build_op(OperationBuilder::new("trait.impl", loc)
        .add_attributes(&[
            (identifier(loc, "sym_name"), string_attr(loc, sym_name)),
            (identifier(loc, "sym_visibility"), string_attr(loc, "private")),
        ])
        .add_regions([declaration_body(loc, &arguments)]))
}

/// Build a `trait.method` named `name` whose type is `function_type`, holding
/// `body`: an empty region for a method a trait requires, or one whose entry
/// block takes the function type's inputs and whose blocks end in
/// `trait.return`. A method carries no visibility: it lives and dies with the
/// trait or impl it stands in.
pub fn method<'c>(loc: Location<'c>,
                  name: &str,
                  function_type: Type<'c>,
                  body: Region<'c>,
) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.method", loc)
        .add_attributes(&[
            (identifier(loc, "sym_name"), string_attr(loc, name)),
            (identifier(loc, "function_type"), type_attr(function_type)),
        ])
        .add_regions([body]))
}

/// Build a `trait.return` of `operands`: the results of the method whose body
/// the block it ends stands in, the evidence an impl returns for its trait's
/// requirements, or the one claim a proof proves.
pub fn return_<'c>(loc: Location<'c>, operands: &[Value<'c, '_>]) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.return", loc).add_operands(operands))
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

/// Create a `trait.project` op selecting requirement `index` of `src_claim`:
/// its trait's requirements in order, then, when the claim is proven by a
/// proof, the where entries of the impl that proof derives it from.
/// `result_claim` spells the claim that selection derives, which verification
/// checks.
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

/// Create a `trait.derive` of `trait_app` from the impl `impl_name` at
/// `impl_args`, the argument of each of the impl's parameters by position, with
/// `premises` holding one claim per entry of the impl's where clause, in its
/// order.
pub fn derive<'c>(loc: Location<'c>,
                  trait_app: TraitApplicationAttribute<'c>,
                  impl_name: &str,
                  impl_args: &[Type<'c>],
                  premises: &[Value<'c,'_>],
) -> Operation<'c> {
    let claim = unproven_claim(loc, trait_app.into());
    build_op(OperationBuilder::new("trait.derive", loc)
        .add_operands(premises)
        .add_attributes(&[
            (identifier(loc, "impl"), symbol_ref_attr(loc, impl_name)),
            (identifier(loc, "impl_args"), type_array_attr(loc, impl_args)),
        ])
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

/// Create a projection-resolution `trait.witness` of `result_type`, an
/// equality claim whose left side is the projection the impl `impl_name`
/// resolves at `impl_args`, the argument of each of its parameters by
/// position, with `premises` holding one claim per entry of that impl's where
/// clause, in its order.
pub fn witness_proj_resolve<'c>(loc: Location<'c>, impl_name: &str, impl_args: &[Type<'c>], premises: &[Value<'c, '_>], result_type: Type<'c>) -> Operation<'c> {
    build_op(OperationBuilder::new("trait.witness", loc)
        .add_attributes(&[
            (identifier(loc, "impl"), symbol_ref_attr(loc, impl_name)),
            (identifier(loc, "impl_args"), type_array_attr(loc, impl_args)),
        ])
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

