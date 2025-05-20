use melior::{ir::{Location, Operation, Type, TypeLike, Value}, Context, pass::Pass, StringRef};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirPass, MlirStringRef, MlirType, MlirValue};

#[link(name = "trait_dialect")]
unsafe extern "C" {
    fn traitRegisterDialect(ctx: MlirContext);
    fn traitCreateMonomorphizePass() -> MlirPass;
    fn traitTraitOpCreate(loc: MlirLocation, name: MlirStringRef) -> MlirOperation;
    fn traitImplOpCreate(loc: MlirLocation, trait_name: MlirStringRef, concrete_type: MlirType) -> MlirOperation;
    fn traitMethodCallOpCreate(loc: MlirLocation,
                               trait_name: MlirStringRef,
                               method_name: MlirStringRef,
                               method_function_type: MlirType,
                               receiver_type: MlirType,
                               arguments: *const MlirValue, num_arguments: isize,
                               result_types: *const MlirType, num_results: isize) -> MlirOperation;
    fn traitFuncCallOpCreate(loc: MlirLocation,
                             callee: MlirStringRef,
                             callee_function_type: MlirType,
                             arguments: *const MlirValue, num_arguments: isize,
                             result_types: *const MlirType, num_results: isize) -> MlirOperation;
    fn traitSelfTypeGet(ctx: MlirContext) -> MlirType;
    fn traitPolyTypeGet(ctx: MlirContext, unique_id: u32,
                        trait_bounds: *const MlirStringRef, num_trait_bounds: isize) -> MlirType;
}

pub fn register(context: &Context) {
    unsafe { traitRegisterDialect(context.to_raw()) }
}

pub fn create_monomorphize_pass() -> Pass {
    unsafe { Pass::from_raw(
        traitCreateMonomorphizePass()
    )}
}

pub fn trait_<'c>(loc: Location<'c>, name: &str) -> Operation<'c> {
    unsafe { Operation::from_raw(traitTraitOpCreate(
        loc.to_raw(),
        StringRef::new(name).to_raw(),
    ))}
}

pub fn impl_<'c>(loc: Location<'c>, trait_name: &str, concrete_type: Type) -> Operation<'c> {
    unsafe { Operation::from_raw(traitImplOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        concrete_type.to_raw(),
    ))}
}

pub fn method_call<'c>(loc: Location<'c>,
                       trait_name: &str,
                       method_name: &str,
                       method_function_type: Type<'c>,
                       receiver_type: Type<'c>,
                       arguments: &[Value<'c,'_>],
                       result_types: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitMethodCallOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        StringRef::new(method_name).to_raw(),
        method_function_type.to_raw(),
        receiver_type.to_raw(),
        arguments.as_ptr() as *const _,
        arguments.len() as isize,
        result_types.as_ptr() as *const _,
        result_types.len() as isize,
    ))}
}

pub fn func_call<'c>(loc: Location<'c>,
                     callee: &str,
                     callee_function_type: Type<'c>,
                     arguments: &[Value<'c,'_>],
                     result_types: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitFuncCallOpCreate(
        loc.to_raw(),
        StringRef::new(callee).to_raw(),
        callee_function_type.to_raw(),
        arguments.as_ptr() as *const _,
        arguments.len() as isize,
        result_types.as_ptr() as *const _,
        result_types.len() as isize,
    ))}
}

pub fn self_type(context: &Context) -> Type {
    unsafe { Type::from_raw(traitSelfTypeGet(context.to_raw())) }
}

pub fn poly_type<'c>(
    context: &'c Context,
    unique_id: u32,
    trait_bounds: &[&str]
) -> Type<'c> {
    let c_refs: Vec<MlirStringRef> = trait_bounds
        .iter()
        .map(|&s| MlirStringRef {
            data: s.as_ptr() as *const _,
            length: s.len() as usize,
        })
        .collect();

    unsafe { Type::from_raw(traitPolyTypeGet(
        context.to_raw(),
        unique_id,
        c_refs.as_ptr(),
        c_refs.len() as isize,
    ))}
}
