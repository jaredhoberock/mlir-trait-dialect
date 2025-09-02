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
    let loc = Location::unknown(&context);
    let mut module = Module::new(loc);

    let i1_ty = IntegerType::new(&context, 1).into();
    let self_ty = trait_::poly_type(&context, 0);
    let other_ty = trait_::poly_type(&context, 1);

    let partial_eq = {
        let vis_id = Identifier::new(&context, "sym_visibility");
        let private_attr = StringAttribute::new(&context, "private").into();

        // (!S, !O) -> i1
        let eq_ty = FunctionType::new(&context, &[self_ty, other_ty], &[i1_ty]).into();
        let eq = func::func(
            &context,
            StringAttribute::new(&context, "eq"),
            TypeAttribute::new(eq_ty),
            Region::new(),
            &[(vis_id, private_attr)],
            loc,
        );

        let neq = {
            // (!S, !O) -> i1
            let neq_ty = FunctionType::new(&context, &[self_ty, other_ty], &[i1_ty]).into();
            let neq = func::func(
                &context,
                StringAttribute::new(&context, "neq"),
                TypeAttribute::new(neq_ty),
                Region::new(),
                &[],
                loc,
            );

            let block = Block::new(&[(self_ty, loc), (other_ty, loc)]);
            let c = block.append_operation(trait_::assume(
                loc,
                "PartialEq",
                &[self_ty, other_ty],
            ));

            let equal = block.append_operation(trait_::method_call(
                loc,
                "PartialEq",
                "eq",
                c.result(0).unwrap().into(),           // claim
                &[
                    block.argument(0).unwrap().into(), // self
                    block.argument(1).unwrap().into(), // other
                ],
                &[i1_ty],
            ));
            let true_ = block.append_operation(arith::constant(
                &context,
                IntegerAttribute::new(i1_ty, 1).into(),
                loc,
            ));
            let result = block.append_operation(arith::xori(
                equal.result(0).unwrap().into(),
                true_.result(0).unwrap().into(),
                loc,
            ));
            block.append_operation(func::r#return(
                &[result.result(0).unwrap().into()],
                loc,
            ));

            neq.regions().next().unwrap()
                .append_block(block);
            neq
        };

        let partial_eq = trait_::trait_(
            loc,
            "PartialEq",
            &[self_ty, other_ty],
        );

        let block = partial_eq.region(0).unwrap().first_block().unwrap();
        block.append_operation(eq);
        block.append_operation(neq);

        partial_eq
    };

    module.body().append_operation(partial_eq);

    let partial_eq_impl_i32_i32 = {
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
                loc,
            );

            let block = Block::new(&[(i32_ty, loc), (i32_ty, loc)]);
            let result = block.append_operation(arith::cmpi(
                &context,
                arith::CmpiPredicate::Eq,
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                loc,
            ));
            block.append_operation(func::r#return(
                &[result.result(0).unwrap().into()],
                loc,
            ));

            eq.regions().next().unwrap()
                .append_block(block);
            eq
        };

        let partial_eq_impl_i32_i32 = trait_::impl_(
            loc,
            "PartialEq",
            &[i32_ty, i32_ty],
        );

        let block = partial_eq_impl_i32_i32
            .region(0)
            .unwrap()
            .first_block()
            .unwrap();
        block.append_operation(eq);

        partial_eq_impl_i32_i32
    };

    module.body().append_operation(partial_eq_impl_i32_i32);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // !T = trait.poly<2>
    // func.func @foo(%c: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
    //   %res = trait.method.call %c @PartialEq[!T,!T]::@eq(%x, %y)
    //     :  (!S,!O) -> i1
    //     as (!T, !T) -> i1
    //   return %res : i1
    // }
    let poly_ty = trait_::poly_type(&context, 2);
    let claim_ty = trait_::claim_type(
        &context,
        "PartialEq",
        &[poly_ty, poly_ty],
    );

    let foo = {
        let foo_ty = FunctionType::new(&context, &[claim_ty, poly_ty, poly_ty], &[i1_ty]).into();
        let foo = func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(foo_ty),
            Region::new(),
            &[],
            loc,
        );

        let block = Block::new(&[(claim_ty, loc), (poly_ty, loc), (poly_ty, loc)]);
        let result = block.append_operation(trait_::method_call(
            loc,
            "PartialEq",
            "eq",
            block.argument(0).unwrap().into(),     // %c
            &[
                block.argument(1).unwrap().into(), // %x
                block.argument(2).unwrap().into(), // %y
            ],
            &[i1_ty],
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            loc,
        ));

        foo.regions().next().unwrap()
            .append_block(block);
        foo
    };

    module.body().append_operation(foo);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // @bar(%x: i32, %y: i32) -> i1 {
    //   %c = trait.allege @PartialEq[i32,i32]
    //   %result = trait.func.call @foo(%c, %x, %y)
    //     : (!P,!T,!T) -> i1
    //     as (!trait.claim<@PartialEq[i32,i32]>,i32,i32) -> i1
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
            loc,
        );

        let block = Block::new(&[(i32_ty, loc), (i32_ty, loc)]);
        let p = block.append_operation(trait_::allege(
            loc,
            "PartialEq",
            &[i32_ty, i32_ty],
        ));
        let result = block.append_operation(trait_::func_call(
            loc,
            "foo",
            &[
                p.result(0).unwrap().into(),
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[i1_ty],
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            loc,
        ));

        bar.regions().next().unwrap()
            .append_block(block);
        bar
    };

    // emit a wrapper function for @bar because we will call it below
    bar.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));

    module.body().append_operation(bar);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // !C = !trait.claim<@PartialEq[!T,!T]>
    // func.func @baz(%c: !C, %x: !T, %y: !T) -> i1 {
    //   %eq = trait.method.call %c @PartialEq[!T,!T]::@eq(%x, %y)
    //     :  (!S,!O) -> i1
    //     as (!T,!T) -> i1
    //   %neq = trait.method.call %c @PartialEq[!T,!T]::@neq(%x, %y)
    //     :  (!S,!O) -> i1
    //     as (!T,!T) -> i1
    //   %result = arith.ori %eq, %neq : i1
    //   return %result : i1
    // }
    let claim_ty = trait_::claim_type(
        &context,
        "PartialEq",
        &[poly_ty, poly_ty],
    );
    let baz = {
        let baz_ty = FunctionType::new(&context, &[claim_ty, poly_ty, poly_ty], &[i1_ty]).into();
        let baz = func::func(
            &context,
            StringAttribute::new(&context, "baz"),
            TypeAttribute::new(baz_ty),
            Region::new(),
            &[],
            loc,
        );

        let block = Block::new(&[(claim_ty, loc), (poly_ty, loc), (poly_ty, loc)]);
        let eq = block.append_operation(trait_::method_call(
            loc,
            "PartialEq",
            "eq",
            block.argument(0).unwrap().into(),     // c
            &[
                block.argument(1).unwrap().into(), // x
                block.argument(2).unwrap().into(), // y
            ],
            &[i1_ty],
        ));
        let neq = block.append_operation(trait_::method_call(
            loc,
            "PartialEq",
            "neq",
            block.argument(0).unwrap().into(),     // c
            &[
                block.argument(1).unwrap().into(), // x
                block.argument(2).unwrap().into(), // y
            ],
            &[i1_ty],
        ));
        let result = block.append_operation(arith::ori(
            eq.result(0).unwrap().into(),
            neq.result(0).unwrap().into(),
            loc,
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            loc,
        ));

        baz.regions().next().unwrap()
            .append_block(block);
        baz
    };

    module.body().append_operation(baz);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // func.func @qux(%x: i32, %y: i32) -> i1 {
    //   %c = trait.allege @PartialEq[i32,i32]
    //   %res = trait.func.call @baz(%c, %x, %y)
    //     : (!C,!T,!T) -> i1
    //     as (!trait.claim<@PartialEq[i32,i32]>, i32,i32) -> i1
    //   return %res : i1
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
            loc,
        );

        let block = Block::new(&[(i32_ty, loc), (i32_ty, loc)]);
        let p = block.append_operation(trait_::allege(
            loc,
            "PartialEq",
            &[i32_ty,i32_ty],
        ));
        let result = block.append_operation(trait_::func_call(
            loc,
            "baz",
            &[
                p.result(0).unwrap().into(),       // c
                block.argument(0).unwrap().into(), // x
                block.argument(1).unwrap().into(), // y
            ],
            &[i1_ty],
        ));
        block.append_operation(func::r#return(
            &[result.result(0).unwrap().into()],
            loc,
        ));

        qux.regions().next().unwrap()
            .append_block(block);
        qux
    };

    // emit a wrapper function for @qux because we will call it below
    qux.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));

    module.body().append_operation(qux);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(trait_::create_monomorphize_pass());
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
