use melior::{
    Context, pass::Pass, StringRef,
    ir::{AttributeLike, Location, Operation, Type, Value, ValueLike},
    ir::attribute::Attribute,
};
use mlir_sys::{
    MlirAttribute, MlirContext, MlirLocation,
    MlirOperation, MlirPass, MlirStringRef,
    MlirType, MlirValue,
};

unsafe extern "C" {
    fn traitRegisterDialect(ctx: MlirContext);
    fn traitCreateMonomorphizePass() -> MlirPass;
    fn traitTraitApplicationAttrGet(ctx: MlirContext,
                                    trait_name: MlirStringRef,
                                    type_args: *const MlirType, num_type_args: isize) -> MlirAttribute;
    fn traitTraitOpCreate(loc: MlirLocation,
                          name: MlirStringRef,
                          type_params: *const MlirType, num_type_params: isize,
                          requirements: *const MlirAttribute, num_requirements: isize) -> MlirOperation;
    fn traitImplOpCreate(loc: MlirLocation,
                         trait_name: MlirStringRef,
                         type_args: *const MlirType, num_type_args: isize,
                         assumptions: *const MlirAttribute, num_assumptions: isize) -> MlirOperation;
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
                           trait_name: MlirStringRef,
                           type_args: *const MlirType, num_type_args: isize) -> MlirOperation;
    fn traitWitnessOpCreate(loc: MlirLocation,
                            proof_name: MlirStringRef,
                            trait_name: MlirStringRef,
                            type_args: *const MlirType, num_type_args: isize) -> MlirOperation;
    fn traitProjectOpCreate(loc: MlirLocation,
                            src_claim: MlirValue,
                            trait_name: MlirStringRef,
                            type_args: *const MlirType, num_type_args: isize) -> MlirOperation;
    fn traitAssumeOpCreate(loc: MlirLocation,
                           trait_app: MlirAttribute) -> MlirOperation;
    fn traitPolyTypeGet(ctx: MlirContext, unique_id: u32) -> MlirType;
    fn traitClaimTypeGet(ctx: MlirContext,
                         trait_name: MlirStringRef,
                         type_args: *const MlirType, num_type_args: isize) -> MlirType;
}

pub fn register(ctx: &Context) {
    unsafe { traitRegisterDialect(ctx.to_raw()) }
}

pub fn create_monomorphize_pass() -> Pass {
    unsafe { Pass::from_raw(
        traitCreateMonomorphizePass()
    )}
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

impl<'c> From<TraitApplicationAttribute<'c>> for Attribute<'c> {
    fn from(a: TraitApplicationAttribute<'c>) -> Self { a.attribute }
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

pub fn impl_<'c>(loc: Location<'c>,
                 trait_name: &str,
                 type_args: &[Type<'c>],
                 assumptions: &[TraitApplicationAttribute<'c>],
) -> Operation<'c> {
    let asm_attrs: Vec<Attribute<'c>> =
        assumptions.iter().copied().map(Into::into).collect();
    unsafe { Operation::from_raw(traitImplOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
        asm_attrs.as_ptr() as *const _,
        asm_attrs.len() as isize,
    ))}
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
                  trait_name: &str,
                  type_args: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitAllegeOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
    ))}
}

pub fn witness<'c>(loc: Location<'c>,
                   proof_name: &str,
                   trait_name: &str,
                   type_args: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitWitnessOpCreate(
        loc.to_raw(),
        StringRef::new(proof_name).to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
    ))}
}

pub fn project<'c>(loc: Location<'c>,
                   src_claim: Value<'c,'_>,
                   trait_name: &str,
                   type_args: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitProjectOpCreate(
        loc.to_raw(),
        src_claim.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
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

pub fn poly_type<'c>(
    ctx: &'c Context,
    unique_id: u32,
) -> Type<'c> {
    unsafe { Type::from_raw(traitPolyTypeGet(
        ctx.to_raw(),
        unique_id,
    ))}
}

pub fn claim_type<'c>(
    ctx: &'c Context,
    trait_name: &str,
    type_args: &[Type<'c>],
) -> Type<'c> {
    unsafe { Type::from_raw(traitClaimTypeGet(
        ctx.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
    ))}
}
