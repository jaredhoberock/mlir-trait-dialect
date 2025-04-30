use trait_dialect as trait_;
use melior::{
    Context,
    dialect::{arith, func, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, RegionLike,
    },
    pass::{self, PassManager},
    utility::{register_all_dialects},
};

#[test]
fn test_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // begin creating a module
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let i1_ty = IntegerType::new(&context, 1).into();
    let self_ty = trait_::self_type(&context);
    let partial_eq = {
        // (!self, !self) -> i1
        let method_ty = FunctionType::new(&context, &[self_ty, self_ty], &[i1_ty]).into();
        let vis_id = Identifier::new(&context, "sym_visibility");
        let private_attr = StringAttribute::new(&context, "private").into();

        let eq = func::func(
            &context,
            StringAttribute::new(&context, "eq"),
            TypeAttribute::new(method_ty),
            Region::new(),
            &[(vis_id, private_attr)],
            location,
        );

        let neq = {
            let neq = func::func(
                &context,
                StringAttribute::new(&context, "neq"),
                TypeAttribute::new(method_ty),
                Region::new(),
                &[],
                location,
            );

            let block = Block::new(&[(self_ty, location), (self_ty, location)]);
            let equal = block.append_operation(trait_::method_call(
                location,
                "PartialEq",
                &self_ty,
                "eq",
                &[
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                ],
                &[i1_ty],
            ));
            let true_ = block.append_operation(arith::constant(
                &context,
                IntegerAttribute::new(i1_ty, 1).into(),
                location,
            ));
            let result = block.append_operation(arith::xori(
                equal.result(0).unwrap().into(),
                true_.result(0).unwrap().into(),
                location,
            ));
            block.append_operation(func::r#return(
                &[result.result(0).unwrap().into()],
                location,
            ));

            neq.regions().next().unwrap()
                .append_block(block);
            neq
        };

        let partial_eq = trait_::trait_(location, "PartialEq");

        let block = Block::new(&[]);
        block.append_operation(eq);
        block.append_operation(neq);

        partial_eq
            .regions()
            .next()
            .unwrap()
            .append_block(block);
        partial_eq
    };

    let partial_eq_impl_i32 = {
        let i32_ty = IntegerType::new(&context, 32).into();
        let eq = {
            // (i32, i32) -> i1
            let method_ty = FunctionType::new(&context, &[i32_ty, i32_ty], &[i1_ty]).into();
            let eq = func::func(
                &context,
                StringAttribute::new(&context, "eq"),
                TypeAttribute::new(method_ty),
                Region::new(),
                &[],
                location,
            );

            let block = Block::new(&[(i32_ty, location), (i32_ty, location)]);
            let result = block.append_operation(arith::cmpi(
                &context,
                arith::CmpiPredicate::Eq,
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));
            block.append_operation(func::r#return(
                &[result.result(0).unwrap().into()],
                location,
            ));

            eq.regions().next().unwrap()
                .append_block(block);
            eq
        };

        let partial_eq_impl_i32 = trait_::impl_(location, "PartialEq", i32_ty);

        let block = Block::new(&[]);
        block.append_operation(eq);

        partial_eq_impl_i32
            .regions()
            .next()
            .unwrap()
            .append_block(block);
        partial_eq_impl_i32
    };

    // !T = trait.poly<0,[@PartialEq]>
    // func.func @foo(%x: !T, %y: !T) -> i1 {
    //   %result = trait.method.call @PartialEq<!T>::@eq(%x, %y) : (!T,!T) -> i1
    //   return %result : i1
    // }
    let foo = {
        let poly_ty = trait_::poly_type(
            &context,
            0,
            &["PartialEq"],
        );

        let foo_ty = FunctionType::new(&context, &[poly_ty, poly_ty], &[i1_ty]).into();

        let foo = func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(foo_ty),
            Region::new(),
            &[],
            location,
        );

        let block = Block::new(&[(poly_ty, location), (poly_ty, location)]);
        let result = block.append_operation(trait_::method_call(
            location,
            "PartialEq",
            &poly_ty,
            "eq",
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i1_ty],
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            location,
        ));

        foo.regions().next().unwrap()
            .append_block(block);
        foo
    };

    // @bar(%x: i32, %y: i32) -> i1 {
    //   %result = trait.func.call @foo(%x, %y) : (i32,i32) -> i1
    //   return %result : i1
    // }
    let mut bar = {
        let i32_ty = IntegerType::new(&context, 32).into();
        let bar_ty = FunctionType::new(&context, &[i32_ty, i32_ty], &[i1_ty]).into();

        let bar = func::func(
            &context,
            StringAttribute::new(&context, "bar"),
            TypeAttribute::new(bar_ty),
            Region::new(),
            &[],
            location,
        );

        let block = Block::new(&[(i32_ty, location), (i32_ty, location)]);
        let result = block.append_operation(trait_::func_call(
            location,
            "foo",
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i1_ty],
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            location,
        ));

        bar.regions().next().unwrap()
            .append_block(block);
        bar
    };

    // emit a wrapper function for @bar because we will call it below
    bar.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));

    // !T = trait.poly<0,[@PartialEq]>
    // func.func @baz(%x: !T, %y: !T) -> i1 {
    //   %eq = trait.method.call @PartialEq<!T>::@eq(%x, %y) : (!T,!T) -> i1
    //   %neq = trait.method.call @PartialEq<!T>::@neq(%x, %y) : (!T,!T) -> i1
    //   %result = arith.ori %eq, %neq : i1
    //   return %result : i1
    // }
    let baz = {
        let poly_ty = trait_::poly_type(
            &context,
            0,
            &["PartialEq"],
        );
        let baz_ty = FunctionType::new(&context, &[poly_ty, poly_ty], &[i1_ty]).into();

        let baz = func::func(
            &context,
            StringAttribute::new(&context, "baz"),
            TypeAttribute::new(baz_ty),
            Region::new(),
            &[],
            location,
        );

        let block = Block::new(&[(poly_ty, location), (poly_ty, location)]);
        let eq = block.append_operation(trait_::method_call(
            location,
            "PartialEq",
            &poly_ty,
            "eq",
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i1_ty],
        ));
        let neq = block.append_operation(trait_::method_call(
            location,
            "PartialEq",
            &poly_ty,
            "neq",
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i1_ty],
        ));
        let result = block.append_operation(arith::ori(
            eq.result(0).unwrap().into(),
            neq.result(0).unwrap().into(),
            location,
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            location,
        ));

        baz.regions().next().unwrap()
            .append_block(block);
        baz
    };

    // func.func @qux(%x: i32, %y: i32) -> i1 {
    //   %result = trait.func.call @baz(%x, %y) : (i32,i32) -> i1
    //   return %result : i1
    // }
    let mut qux = {
        let i32_ty = IntegerType::new(&context, 32).into();
        let qux_ty = FunctionType::new(&context, &[i32_ty, i32_ty], &[i1_ty]).into();

        let qux = func::func(
            &context,
            StringAttribute::new(&context, "qux"),
            TypeAttribute::new(qux_ty),
            Region::new(),
            &[],
            location,
        );

        let block = Block::new(&[(i32_ty, location), (i32_ty, location)]);
        let result = block.append_operation(trait_::func_call(
            location,
            "baz",
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i1_ty],
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            location,
        ));

        qux.regions().next().unwrap()
            .append_block(block);
        qux
    };

    // emit a wrapper function for @qux because we will call it below
    qux.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));

    module.body().append_operation(partial_eq);
    module.body().append_operation(partial_eq_impl_i32);
    module.body().append_operation(foo);
    module.body().append_operation(bar);
    module.body().append_operation(baz);
    module.body().append_operation(qux);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test that we can call bar & qux and they produce the expected results

    unsafe {
        // @bar is equivalent to a function that compares its arguments and returns whether or not they are equal

        {
            let mut x: i32 = 7;
            let mut y: i32 = 13;
            let mut result: bool = true;

            let mut packed_args: [*mut (); 3] = [
                &mut x as *mut i32 as *mut (),
                &mut y as *mut i32 as *mut (),
                &mut result as *mut bool as *mut (),
            ];

            engine.invoke_packed("bar", &mut packed_args)
                .expect("JIT invocation failed");

            assert_eq!(result, false);
        }

        {
            let mut x: i32 = 7;
            let mut y: i32 = 7;
            let mut result: bool = false;

            let mut packed_args: [*mut (); 3] = [
                &mut x as *mut i32 as *mut (),
                &mut y as *mut i32 as *mut (),
                &mut result as *mut bool as *mut (),
            ];

            engine.invoke_packed("bar", &mut packed_args)
                .expect("JIT invocation failed");

            assert_eq!(result, true);
        }
    }

    unsafe {
        // @qux should always return true whether or not its arguments are equal
        
        {
            let mut x: i32 = 7;
            let mut y: i32 = 13;
            let mut result: bool = false;

            let mut packed_args: [*mut (); 3] = [
                &mut x as *mut i32 as *mut (),
                &mut y as *mut i32 as *mut (),
                &mut result as *mut bool as *mut (),
            ];

            engine.invoke_packed("qux", &mut packed_args)
                .expect("JIT invocation failed");

            assert_eq!(result, true);
        }

        {
            let mut x: i32 = 7;
            let mut y: i32 = 7;
            let mut result: bool = false;

            let mut packed_args: [*mut (); 3] = [
                &mut x as *mut i32 as *mut (),
                &mut y as *mut i32 as *mut (),
                &mut result as *mut bool as *mut (),
            ];

            engine.invoke_packed("qux", &mut packed_args)
                .expect("JIT invocation failed");

            assert_eq!(result, true);
        }
    }
}
