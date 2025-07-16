use melior::{ir::{Location, Operation, Type, TypeLike, Value, ValueLike}, Context, pass::Pass, StringRef};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirPass, MlirStringRef, MlirType, MlirValue};

unsafe extern "C" {
    fn traitRegisterDialect(ctx: MlirContext);
    fn traitCreateMonomorphizePass() -> MlirPass;
    fn traitTraitOpCreate(loc: MlirLocation,
                          name: MlirStringRef,
                          type_params: *const MlirType, num_type_params: isize) -> MlirOperation;
    fn traitImplOpCreate(loc: MlirLocation,
                         trait_name: MlirStringRef,
                         type_args: *const MlirType, num_type_args: isize) -> MlirOperation;
    fn traitMethodCallOpCreate(loc: MlirLocation,
                               trait_name: MlirStringRef,
                               method_name: MlirStringRef,
                               method_function_type: MlirType,
                               witness: MlirValue,
                               arguments: *const MlirValue, num_arguments: isize,
                               result_types: *const MlirType, num_results: isize) -> MlirOperation;
    fn traitFuncCallOpCreate(loc: MlirLocation,
                             callee: MlirStringRef,
                             callee_function_type: MlirType,
                             arguments: *const MlirValue, num_arguments: isize,
                             result_types: *const MlirType, num_results: isize) -> MlirOperation;
    fn traitWitnessOpCreate(loc: MlirLocation,
                            trait_name: MlirStringRef,
                            type_args: *const MlirType, num_type_args: isize) -> MlirOperation;
    fn traitPolyTypeGet(ctx: MlirContext, unique_id: u32) -> MlirType;
    fn traitWitnessTypeGet(ctx: MlirContext,
                           trait_name: MlirStringRef,
                           type_args: *const MlirType, num_type_args: isize) -> MlirType;
}

pub fn register(context: &Context) {
    unsafe { traitRegisterDialect(context.to_raw()) }
}

pub fn create_monomorphize_pass() -> Pass {
    unsafe { Pass::from_raw(
        traitCreateMonomorphizePass()
    )}
}

pub fn trait_<'c>(loc: Location<'c>,
                  name: &str,
                  type_params: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitTraitOpCreate(
        loc.to_raw(),
        StringRef::new(name).to_raw(),
        type_params.as_ptr() as *const _,
        type_params.len() as isize,
    ))}
}

pub fn impl_<'c>(loc: Location<'c>,
                 trait_name: &str,
                 type_args: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitImplOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
    ))}
}

pub fn method_call<'c>(loc: Location<'c>,
                       trait_name: &str,
                       method_name: &str,
                       method_function_type: Type<'c>,
                       witness: Value<'c,'_>,
                       arguments: &[Value<'c,'_>],
                       result_types: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitMethodCallOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        StringRef::new(method_name).to_raw(),
        method_function_type.to_raw(),
        witness.to_raw(),
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

pub fn witness<'c>(loc: Location<'c>,
                   trait_name: &str,
                   type_args: &[Type<'c>],
) -> Operation<'c> {
    unsafe { Operation::from_raw(traitWitnessOpCreate(
        loc.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
    ))}
}

pub fn poly_type<'c>(
    context: &'c Context,
    unique_id: u32,
) -> Type<'c> {
    unsafe { Type::from_raw(traitPolyTypeGet(
        context.to_raw(),
        unique_id,
    ))}
}

pub fn witness_type<'c>(
    context: &'c Context,
    trait_name: &str,
    type_args: &[Type<'c>],
) -> Type<'c> {
    unsafe { Type::from_raw(traitWitnessTypeGet(
        context.to_raw(),
        StringRef::new(trait_name).to_raw(),
        type_args.as_ptr() as *const _,
        type_args.len() as isize,
    ))}
}
