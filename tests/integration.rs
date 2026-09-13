// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use trait_dialect as trait_;
use melior::{
    Context,
    dialect::{arith, func, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        operation::{OperationLike, OperationMutLike},
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
                trait_::claim_type(
                  &context,
                  trait_::trait_application_attr(
                    &context,
                    "PartialEq",
                    &[self_ty, other_ty],
                  ),
                ).into(),
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
                // the method binds no type variables of its own
                &[],
                &[],
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
            &[], // no where-clause predicates
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
            trait_::trait_application_attr(
                &context,
                "PartialEq",
                &[i32_ty, i32_ty],
            ),
            &[], // no assumptions
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
        trait_::trait_application_attr(
          &context,
          "PartialEq",
          &[poly_ty, poly_ty],
        ),
    ).into();

    let foo = {
        // a polymorphic function is a template, and a template is private from
        // birth so that nothing outside its own symbol table may name it once
        // its instances are cut
        let vis_id = Identifier::new(&context, "sym_visibility");
        let private_attr = StringAttribute::new(&context, "private").into();

        let foo_ty = FunctionType::new(&context, &[claim_ty, poly_ty, poly_ty], &[i1_ty]).into();
        let foo = func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(foo_ty),
            Region::new(),
            &[(vis_id, private_attr)],
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
            // the method binds no type variables of its own
            &[],
            &[],
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
            trait_::trait_application_attr(
                &context,
                "PartialEq",
                &[i32_ty, i32_ty],
            ),
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
            // the callee's own type parameter, and the argument this call
            // supplies for it
            &[trait_::poly_type(&context, 2)],
            &[i32_ty],
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
        trait_::trait_application_attr(
            &context,
            "PartialEq",
            &[poly_ty, poly_ty],
        ),
    ).into();

    let baz = {
        // a polymorphic function is a template, and a template is private from
        // birth so that nothing outside its own symbol table may name it once
        // its instances are cut
        let vis_id = Identifier::new(&context, "sym_visibility");
        let private_attr = StringAttribute::new(&context, "private").into();

        let baz_ty = FunctionType::new(&context, &[claim_ty, poly_ty, poly_ty], &[i1_ty]).into();
        let baz = func::func(
            &context,
            StringAttribute::new(&context, "baz"),
            TypeAttribute::new(baz_ty),
            Region::new(),
            &[(vis_id, private_attr)],
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
            // the method binds no type variables of its own
            &[],
            &[],
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
            // the method binds no type variables of its own
            &[],
            &[],
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
            trait_::trait_application_attr(
                &context,
                "PartialEq",
                &[i32_ty,i32_ty],
            ),
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
            // the callee's own type parameter, and the argument this call
            // supplies for it
            &[trait_::poly_type(&context, 2)],
            &[i32_ty],
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
    // monomorphization is two passes: instantiate the monomorphs the calls need,
    // then erase the residual polymorphism and collect the templates nothing
    // names.
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(trait_::create_instantiate_monomorphs_pass());
    pass_manager.add_pass(trait_::create_erase_polymorphs_pass());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false, false);

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

#[test]
fn the_project_builder_selects_a_requirement_by_position() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let loc = Location::unknown(&context);
    let module = Module::new(loc);
    let self_ty = trait_::poly_type(&context, 0);
    let i32_ty: melior::ir::Type = IntegerType::new(&context, 32).into();
    let i64_ty: melior::ir::Type = IntegerType::new(&context, 64).into();

    // @Has requires @A[Self] at position 0 and Self::Out = i64 at position 1.
    module
        .body()
        .append_operation(trait_::trait_(loc, "A", &[self_ty], &[]));

    let has_self = trait_::trait_application_attr(&context, "Has", &[self_ty]);
    let a_self = trait_::trait_application_attr(&context, "A", &[self_ty]);
    let out_of_self = trait_::projection_type(&context, has_self, "Out", &[]);
    let out_is_i64 = trait_::type_equality_attr(&context, out_of_self, i64_ty)
        .expect("the equality requirement constructs");
    let has = trait_::trait_(loc, "Has", &[self_ty], &[a_self.into(), out_is_i64]);
    has.region(0)
        .unwrap()
        .first_block()
        .unwrap()
        .append_operation(trait_::assoc_type(loc, "Out", None, &[]));
    module.body().append_operation(has);

    // A hop built at position 1 off a claim of @Has[i32] is that equality
    // requirement instantiated at i32: the builder writes the index the
    // verifier reads.
    let has_i32 = trait_::trait_application_attr(&context, "Has", &[i32_ty]);
    let claim_ty: melior::ir::Type = trait_::claim_type(&context, has_i32).into();
    let out_of_i32 = trait_::projection_type(&context, has_i32, "Out", &[]);
    let hop = trait_::equality_claim_type(&context, out_of_i32, i64_ty)
        .expect("the hop claim constructs");

    let block = Block::new(&[(claim_ty, loc)]);
    block.append_operation(trait_::project(
        loc,
        block.argument(0).unwrap().into(),
        1,
        hop,
    ));
    block.append_operation(func::r#return(&[], loc));
    let body = Region::new();
    body.append_block(block);
    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "f"),
        TypeAttribute::new(FunctionType::new(&context, &[claim_ty], &[]).into()),
        body,
        &[],
        loc,
    ));

    assert!(module.as_operation().verify());
    let rendered = module.as_operation().to_string();
    assert!(
        rendered.contains("trait.project %arg0[1]"),
        "the hop selects requirement 1: {rendered}"
    );
}

#[test]
fn the_obligation_aware_verification_demands_the_cited_impl_s_assumptions() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    // @Has_tuple binds @Has[tuple<!U>]::Out to i64 and requires @X[!U]. The
    // module carries no impl of @X, so for !U = i32 the assumption is unmet.
    let module = Module::parse(
        &context,
        "!U = !trait.poly<0>\n\
         trait.trait private @X[!U] {}\n\
         trait.trait private @Has[!U] { trait.assoc_type @Out }\n\
         trait.impl private @Has_tuple for @Has[tuple<!U>] where [@X[!U]] { trait.assoc_type @Out = i64 }\n",
    )
    .expect("the fixture module parses");

    let projection = melior::ir::Type::parse(&context, "!trait.proj<@Has[tuple<i32>], \"Out\">")
        .expect("the projection parses");
    let i64_ty: melior::ir::Type = IntegerType::new(&context, 64).into();
    let i32_ty: melior::ir::Type = IntegerType::new(&context, 32).into();

    // With no premise verification refuses: the impl's @X[i32] assumption is
    // undischarged.
    assert!(!trait_::projection_resolution_verifies_at_use(
        &module, projection, i64_ty, "Has_tuple", &[]
    ));

    // Supplying an @X[i32] application premise discharges the assumption, and
    // verification accepts.
    let x_i32 = trait_::trait_application_attr(&context, "X", &[i32_ty]);
    let x_i32_claim: melior::ir::Type = trait_::claim_type(&context, x_i32).into();
    assert!(trait_::projection_resolution_verifies_at_use(
        &module, projection, i64_ty, "Has_tuple", &[x_i32_claim]
    ));
}

#[test]
fn the_two_monomorphization_steps_render_through_discovery() {
    // The trait dialect contributes its lowering as two steps; the driver
    // discovers them over a context the dialect is registered in. A step is begun
    // with the pass constructor that adds its passes and the legality that decides
    // its readiness, so what the roster carries beside the label is the dialect
    // that contributed it, whether the cleanup interlude follows it, and what it
    // seals: instantiate asks for no cleanup, erase asks for it, and neither seals
    // an operation. Neither step names an operation anywhere on its descriptor --
    // what orders the two is each step's own legality, read against the module.
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let roster = lowering_driver::discover_roster(context.to_raw());
    assert_eq!(roster.step_count(), 2, "the trait dialect contributes exactly two steps");

    let labels: Vec<String> = (0..roster.step_count()).map(|s| roster.step_label(s)).collect();
    let instantiate = labels
        .iter()
        .position(|label| label == "instantiate-monomorphs")
        .expect("instantiate-monomorphs is contributed");
    let erase = labels
        .iter()
        .position(|label| label == "erase-polymorphs")
        .expect("erase-polymorphs is contributed");

    assert_eq!(roster.step_namespace(instantiate), "trait");
    assert_eq!(roster.step_namespace(erase), "trait");
    assert!(!roster.step_wants_cleanup(instantiate));
    assert!(roster.step_wants_cleanup(erase));

    // the roster renders one line per step, field for field: the cleanup request
    // and the seals, with no operation named -- neither step spells its readiness
    // as a vocabulary, and neither seals anything.
    let render = roster.render();
    assert_eq!(render.lines().count(), 2, "{render}");
    assert!(render.contains("instantiate-monomorphs | wants-cleanup=0 | seals:\n"), "{render}");
    assert!(render.contains("erase-polymorphs | wants-cleanup=1 | seals:\n"), "{render}");
}

/// The readiness the driver derives from `module` at the entry boundary,
/// rendered: the roster composed over the loaded dialects, then every step's own
/// ConversionTarget and TypeConverter read against the module, the audit stopping
/// the run there so no pass runs. A step is named in the rendering exactly when it
/// is present -- among the winners when it is ready, in the held census with the
/// reasons it waits when it is not. The permutation is the identity, which spells
/// the order the winners are rendered in and not which steps are ready.
fn entry_readiness(context: &Context, module: &Module) -> String {
    let mut rendered = String::new();
    let (run, _roster) = unsafe {
        lowering_driver::compose_to_target(
            module.as_operation().to_raw(),
            context.to_raw(),
            0,
            std::ptr::null_mut(),
            |_boundary, _label, selection, _decision| {
                rendered = selection.to_string();
                false
            },
        )
    };
    assert_eq!(
        run.outcome,
        lowering_driver::Outcome::Stopped,
        "the audit stops the run at the entry boundary"
    );
    rendered
}

/// The steps ready to run at the boundary `readiness` renders.
fn ready_steps(readiness: &str) -> Vec<String> {
    match lowering_driver::parse_selection(readiness) {
        lowering_driver::Selection::Batch { winners, .. } => winners,
        other => panic!("expected a batch at the entry boundary, got {other:?}"),
    }
}

#[test]
fn instantiate_is_ready_only_while_a_rewritable_call_stands() {
    // The legality instantiate hands the driver marks a pending operation illegal
    // and has an opinion on every other one, so the step is ready exactly where a
    // lowering pattern would fire and waits on nothing. Over a module with one
    // rewritable method call at module scope and one standing inside a template,
    // instantiate is ready: the module-scope call is its work, and the
    // template-interior call -- foreign, what leaves with the template -- is not.
    // erase is held there: its own target has no opinion on the pending call, which
    // is what keeps it behind instantiate. After the pass has instantiated the
    // rewritable call the template-interior call still stands, yet instantiate is no
    // longer present on the module: its target now finds every standing operation
    // legal, and the templates are erase's to take.
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let source = "\
!S = !trait.poly<0>\n\
!V = !trait.poly<9>\n\
trait.trait private @Store[!S] {\n\
  func.func private @keep(!S, !V) -> !V\n\
}\n\
trait.impl private @Store_impl_i64 for @Store[i64] {\n\
  func.func @keep(%self: i64, %v: !trait.poly<5>) -> !trait.poly<5> {\n\
    return %v : !trait.poly<5>\n\
  }\n\
}\n\
func.func private @tpl(%p: !trait.claim<@Store[i64]>, %x: i64, %v: !trait.poly<7>) -> !trait.poly<7> {\n\
  %r = trait.method.call %p @Store[i64]::@keep(%x, %v) : (i64, !trait.poly<7>) -> !trait.poly<7> attributes {type_params = [!trait.poly<9>], type_args = [!trait.poly<7>]}\n\
  return %r : !trait.poly<7>\n\
}\n\
func.func @host(%x: i64, %v: i32) -> i32 {\n\
  %p = trait.witness @Store_impl_i64 for @Store[i64]\n\
  %r = trait.method.call %p @Store[i64]::@keep(%x, %v) : (i64, i32) -> i32 by @Store_impl_i64 attributes {type_params = [!trait.poly<9>], type_args = [i32]}\n\
  return %r : i32\n\
}\n";
    let mut module = Module::parse(&context, source).expect("the fixture module parses");
    assert!(module.as_operation().verify(), "the fixture module verifies");

    let before = entry_readiness(&context, &module);
    assert!(
        ready_steps(&before).iter().any(|step| step == "instantiate-monomorphs"),
        "the module-scope call is instantiate's work, so the step is ready: {before}"
    );
    assert!(
        before.contains("erase-polymorphs:waits(trait.method.call)"),
        "erase has no opinion on the pending call, which is what holds it behind instantiate \
         with no step naming the other's vocabulary: {before}"
    );

    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(trait_::create_instantiate_monomorphs_pass());
    assert!(pass_manager.run(&mut module).is_ok(), "instantiate runs");

    let after = entry_readiness(&context, &module);
    assert!(
        !after.contains("instantiate-monomorphs"),
        "the rewritable call is instantiated and the template interior is not instantiate's \
         work, so the step is no longer present: {after}"
    );
}

#[test]
fn instantiate_is_ready_on_a_standing_claim_obligation() {
    // instantiate's legality marks every operation carrying a standing obligation
    // illegal, not only the calls, so a program whose only pending trait work is an
    // unproven monomorphic claim makes the step ready rather than stranding the run
    // at a boundary no step owns. Here the method call's receiver claim is an
    // unproven allege, so the call is not yet rewritable; the standing allege is
    // what carries the obligation. After the pass has proved the claim and
    // instantiated the call, instantiate is no longer present on the module.
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let source = "\
trait.trait private @T[!trait.poly<0>] {\n\
  func.func private @m(!trait.poly<0>) -> i32\n\
}\n\
trait.impl private @T_i32 for @T[i32] {\n\
  func.func @m(%a: i32) -> i32 {\n\
    %c = arith.constant 1 : i32\n\
    return %c : i32\n\
  }\n\
}\n\
func.func @host(%x: i32) -> i32 {\n\
  %ev = trait.allege @T[i32]\n\
  %r = trait.method.call %ev @T[i32]::@m(%x) : (i32) -> i32\n\
  return %r : i32\n\
}\n";
    let mut module = Module::parse(&context, source).expect("the fixture module parses");

    let before = entry_readiness(&context, &module);
    assert!(
        ready_steps(&before).iter().any(|step| step == "instantiate-monomorphs"),
        "the standing allege is instantiate's work, so the step is ready: {before}"
    );

    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(trait_::create_instantiate_monomorphs_pass());
    assert!(pass_manager.run(&mut module).is_ok(), "instantiate runs");

    let after = entry_readiness(&context, &module);
    assert!(
        !after.contains("instantiate-monomorphs"),
        "instantiate proved the claim and lowered the call, so the step is no longer present: \
         {after}"
    );
}
