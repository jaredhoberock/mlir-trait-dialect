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
        // (!S, !O) -> i1
        let eq_ty = FunctionType::new(&context, &[self_ty, other_ty], &[i1_ty]).into();
        let eq = trait_::method(loc, "eq", eq_ty, Region::new());

        let neq = {
            // (!S, !O) -> i1
            let neq_ty = FunctionType::new(&context, &[self_ty, other_ty], &[i1_ty]).into();
            let neq = trait_::method(loc, "neq", neq_ty, Region::new());

            let block = Block::new(&[(self_ty, loc), (other_ty, loc)]);
            let c = block.append_operation(trait_::assume_self(
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
            block.append_operation(trait_::return_(
                loc,
                &[result.result(0).unwrap().into()],
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
            let eq = trait_::method(loc, "eq", method_ty, Region::new());

            let block = Block::new(&[(i32_ty, loc), (i32_ty, loc)]);
            let result = block.append_operation(arith::cmpi(
                &context,
                arith::CmpiPredicate::Eq,
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                loc,
            ));
            block.append_operation(trait_::return_(
                loc,
                &[result.result(0).unwrap().into()],
            ));

            eq.regions().next().unwrap()
                .append_block(block);
            eq
        };

        let partial_eq_impl_i32_i32 = trait_::impl_named(
            loc,
            "PartialEq_impl_i32_i32",
            trait_::trait_application_attr(
                &context,
                "PartialEq",
                &[i32_ty, i32_ty],
            ),
            &[], // no where-clause predicates
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
            ).into(),
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
            trait_::trait_application_attr(
                &context,
                "PartialEq",
                &[i32_ty,i32_ty],
            ).into(),
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
  %r = trait.method.call %p @Store[i64]::@keep(%x, %v) : (i64, !trait.poly<7>) -> !trait.poly<7>\n\
  return %r : !trait.poly<7>\n\
}\n\
func.func @host(%x: i64, %v: i32) -> i32 {\n\
  %p = trait.witness @Store_impl_i64 for @Store[i64]\n\
  %r = trait.method.call %p @Store[i64]::@keep(%x, %v) : (i64, i32) -> i32 by @Store_impl_i64\n\
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

/// The method body of `@A_gen`, the last operation of `module`: a positional
/// assume is placed where it cites that impl's entries.
fn impl_method_body<'c, 'a>(module: &'a Module<'c>) -> melior::ir::BlockRef<'c, 'a> {
    let mut last = module.body().first_operation().expect("the fixture holds operations");
    while let Some(next) = last.next_in_block() {
        last = next;
    }
    let method = last
        .region(0)
        .expect("an impl has a body")
        .first_block()
        .expect("an impl body has a block")
        .first_operation()
        .expect("the impl holds its method");
    method
        .region(0)
        .expect("a method has a body")
        .first_block()
        .expect("the method body has a block")
}

#[test]
fn the_positional_assume_builders_cite_the_entries_the_verifier_reads() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    // @A_gen's where clause states @B[T] at position 0 and @C[T] at position 1.
    let source = "\
trait.trait private @B[!trait.poly<0>] { func.func private @b(!trait.poly<0>) -> i64 }\n\
trait.trait private @C[!trait.poly<0>] { func.func private @c(!trait.poly<0>) -> i64 }\n\
trait.trait private @A[!trait.poly<0>] { func.func private @a(!trait.poly<0>) -> i64 }\n\
trait.impl private @A_gen for @A[!trait.poly<0>] where [@B[!trait.poly<0>], @C[!trait.poly<0>]] {\n\
  func.func @a(%x: !trait.poly<0>) -> i64 {\n\
    %c = arith.constant 0 : i64\n\
    return %c : i64\n\
  }\n\
}\n";
    let loc = Location::unknown(&context);
    let t = trait_::poly_type(&context, 0);
    let claim = |name: &str| -> melior::ir::Type {
        trait_::claim_type(&context, trait_::trait_application_attr(&context, name, &[t])).into()
    };

    // Each builder writes the entry the verifier reads, and the result type
    // spells the claim that entry states.
    let module = Module::parse(&context, source).expect("the fixture module parses");
    let body = impl_method_body(&module);
    body.insert_operation(0, trait_::assume_self(loc, claim("A")));
    body.insert_operation(1, trait_::assume_entry(loc, 1, claim("C")));
    assert!(module.as_operation().verify());
    let rendered = module.as_operation().to_string();
    assert!(
        rendered.contains("trait.assume self : !trait.claim<@A[!trait.poly<0>]>")
            && rendered.contains("trait.assume 1 : !trait.claim<@C[!trait.poly<0>]>"),
        "the citations print as position and claim: {rendered}"
    );

    // A claim other than the entry at the position is refused.
    let module = Module::parse(&context, source).expect("the fixture module parses");
    impl_method_body(&module).insert_operation(0, trait_::assume_entry(loc, 0, claim("C")));
    assert!(!module.as_operation().verify());
}

/// `forall X where Marker[X] -> Marker[Has[receiver]::A<X>]`, the bound of
/// `type A<X>: Marker where X: Marker` at `receiver`.
fn marker_bound_of_has<'c>(
    context: &'c Context,
    receiver: melior::ir::Type<'c>,
) -> melior::ir::attribute::Attribute<'c> {
    let x = trait_::bound_var_type(context, 0);
    let has = trait_::trait_application_attr(context, "Has", &[receiver]);
    let a_of_x = trait_::projection_type(context, has, "A", &[x]);
    trait_::bound_predicate_attr(
        context,
        1,
        &[trait_::trait_application_attr(context, "Marker", &[x]).into()],
        trait_::trait_application_attr(context, "Marker", &[a_of_x]).into(),
    )
    .expect("the bound predicate constructs")
}

#[test]
fn the_bound_builders_state_and_select_a_quantified_requirement() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let loc = Location::unknown(&context);
    let s = trait_::poly_type(&context, 0);
    let x = trait_::poly_type(&context, 1);
    let t = trait_::poly_type(&context, 2);
    let i1: melior::ir::Type = IntegerType::new(&context, 1).into();
    let i32_ty: melior::ir::Type = IntegerType::new(&context, 32).into();

    // @Has states `forall X where Marker[X] -> Marker[Has[S]::A<X>]`, the bound
    // of `type A<X>: Marker where X: Marker`.
    let bound_at = |receiver| marker_bound_of_has(&context, receiver);
    let module = Module::new(loc);
    module.body().append_operation(trait_::trait_(loc, "Marker", &[s], &[]));
    let has = trait_::trait_(loc, "Has", &[s], &[bound_at(s)]);
    has.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", None, &[x]));
    module.body().append_operation(has);

    // `impl Has for i32 { type A<X> = X; }` proves the bound by its premise.
    let has_i32 = trait_::trait_application_attr(&context, "Has", &[i32_ty]);
    let evidence = |body| {
        trait_::requirement_witness_attr(
            &context, 0,
            trait_::witness_body_attr(&context, body).expect("the body constructs"),
        )
        .expect("the witness constructs")
    };
    let impl_op = trait_::impl_named(loc, "Has_i32", has_i32, &[]);
    impl_op.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", Some(x), &[x]));
    trait_::set_impl_witnesses(&impl_op, &[evidence(trait_::WitnessBody::BinderPremise(0))]);
    module.body().append_operation(impl_op);

    // A generic function selects the requirement at i1 with a claim of its
    // premise there.
    let has_t = trait_::trait_application_attr(&context, "Has", &[t]);
    let has_t_claim: melior::ir::Type = trait_::claim_type(&context, has_t).into();
    let marker_i1: melior::ir::Type = trait_::claim_type(
        &context, trait_::trait_application_attr(&context, "Marker", &[i1])).into();
    let a_of_i1 = trait_::projection_type(&context, has_t, "A", &[i1]);
    let conclusion: melior::ir::Type = trait_::claim_type(
        &context, trait_::trait_application_attr(&context, "Marker", &[a_of_i1])).into();
    let block = Block::new(&[(has_t_claim, loc), (marker_i1, loc)]);
    block.append_operation(trait_::project_bound(
        loc,
        block.argument(0).unwrap().into(),
        0,
        &[i1],
        &[block.argument(1).unwrap().into()],
        conclusion,
    ));
    block.append_operation(func::r#return(&[], loc));
    let body = Region::new();
    body.append_block(block);
    let vis_id = Identifier::new(&context, "sym_visibility");
    let private_attr = StringAttribute::new(&context, "private").into();
    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "f"),
        TypeAttribute::new(FunctionType::new(&context, &[has_t_claim, marker_i1], &[]).into()),
        body,
        &[(vis_id, private_attr)],
        loc,
    ));
    assert!(module.as_operation().verify());
    let rendered = module.as_operation().to_string();
    assert!(
        rendered.contains("by premise 0") && rendered.contains("[0] for [i1] given("),
        "the evidence and the hop print what they state: {rendered}"
    );

    // Evidence citing a where-clause entry the impl does not have is refused.
    let impl_op = trait_::impl_named(loc, "Has_i32", has_i32, &[]);
    impl_op.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", Some(x), &[x]));
    trait_::set_impl_witnesses(&impl_op, &[evidence(trait_::WitnessBody::ImplPremise(0))]);
    let module = Module::new(loc);
    module.body().append_operation(trait_::trait_(loc, "Marker", &[s], &[]));
    let has = trait_::trait_(loc, "Has", &[s], &[bound_at(s)]);
    has.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", None, &[x]));
    module.body().append_operation(has);
    module.body().append_operation(impl_op);
    assert!(!module.as_operation().verify());
}

#[test]
fn the_body_builders_state_a_requirement_hop_and_an_allegation() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let loc = Location::unknown(&context);
    let s = trait_::poly_type(&context, 0);
    let x = trait_::poly_type(&context, 1);
    let p = trait_::poly_type(&context, 2);
    let i64_ty: melior::ir::Type = IntegerType::new(&context, 64).into();
    let x_bound = trait_::bound_var_type(&context, 0);

    // @Has states `forall X -> Marker[Has[S]::A<X>]`; `@Sub` requires `Marker`.
    let bound_at = |receiver| {
        let has = trait_::trait_application_attr(&context, "Has", &[receiver]);
        let a_of_x = trait_::projection_type(&context, has, "A", &[x_bound]);
        trait_::bound_predicate_attr(
            &context,
            1,
            &[],
            trait_::trait_application_attr(&context, "Marker", &[a_of_x]).into(),
        )
        .expect("the bound predicate constructs")
    };
    let module = Module::new(loc);
    module.body().append_operation(trait_::trait_(loc, "Marker", &[s], &[]));
    module.body().append_operation(trait_::trait_(
        loc,
        "Sub",
        &[s],
        &[trait_::trait_application_attr(&context, "Marker", &[s]).into()],
    ));
    let has = trait_::trait_(loc, "Has", &[s], &[bound_at(s)]);
    has.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", None, &[x]));
    module.body().append_operation(has);

    let evidence = |body| {
        trait_::requirement_witness_attr(
            &context, 0,
            trait_::witness_body_attr(&context, body).expect("the body constructs"),
        )
        .expect("the witness constructs")
    };

    // `impl<P: Sub> Has for (P,) { type A<X> = P; }` reads its bound off its
    // premise's requirement.
    let tuple_p: melior::ir::Type = melior::ir::r#type::TupleType::new(&context, &[p]).into();
    let has_tuple = trait_::trait_application_attr(&context, "Has", &[tuple_p]);
    let sub_p = trait_::trait_application_attr(&context, "Sub", &[p]);
    let impl_op = trait_::impl_named(loc, "Has_sub", has_tuple, &[sub_p.into()]);
    impl_op.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", Some(p), &[x]));
    let where_0 = trait_::witness_body_attr(&context, trait_::WitnessBody::ImplPremise(0))
        .expect("the body constructs");
    trait_::set_impl_witnesses(&impl_op, &[evidence(trait_::WitnessBody::RequirementHop {
        position: 0,
        of: where_0,
        type_args: vec![],
        premises: vec![],
    })]);
    module.body().append_operation(impl_op);

    // `impl Has for i64 { type A<X> = i64; }` alleges `Marker[i64]`.
    let has_i64 = trait_::trait_application_attr(&context, "Has", &[i64_ty]);
    let impl_op = trait_::impl_named(loc, "Has_i64", has_i64, &[]);
    impl_op.region(0).unwrap().first_block().unwrap()
        .append_operation(trait_::assoc_type(loc, "A", Some(i64_ty), &[x]));
    let marker_i64 = trait_::trait_application_attr(&context, "Marker", &[i64_ty]);
    trait_::set_impl_witnesses(&impl_op, &[evidence(trait_::WitnessBody::Allegation(marker_i64))]);
    module.body().append_operation(impl_op);

    assert!(module.as_operation().verify());
    let rendered = module.as_operation().to_string();
    assert!(
        rendered.contains("by requirement 0 of where 0") && rendered.contains("by allege @Marker[i64]"),
        "the hop and the allegation print what they state: {rendered}"
    );
}

#[test]
fn the_builders_state_an_impls_arguments_on_a_derive_and_a_proof() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    // @Tr_tuple is an impl of @Tr[tuple<U>] where @Tr[U] and Tr[U]::Out = i64.
    let source = "\
trait.trait private @Tr[!trait.poly<0>] { trait.assoc_type @Out }\n\
trait.impl private @Tr_i32 for @Tr[i32] { trait.assoc_type @Out = i64 }\n\
trait.impl private @Tr_tuple for @Tr[tuple<!trait.poly<1>>] where [@Tr[!trait.poly<1>], !trait.proj<@Tr[!trait.poly<1>], \"Out\"> = i64] {\n\
  trait.assoc_type @Out = i64\n\
}\n";
    let module = Module::parse(&context, source).expect("the fixture module parses");
    let loc = Location::unknown(&context);
    let t = trait_::poly_type(&context, 0);
    let u = trait_::poly_type(&context, 1);
    let i32_ty: melior::ir::Type = IntegerType::new(&context, 32).into();
    let i64_ty: melior::ir::Type = IntegerType::new(&context, 64).into();
    let tuple_of = |ty| melior::ir::r#type::TupleType::new(&context, &[ty]).into();

    // The proof at i32: the trait states no requirement, and the impl's two
    // entries are the application @Tr_i32 discharges and the equality.
    let tr_tuple_i32 = trait_::trait_application_attr(&context, "Tr", &[tuple_of(i32_ty)]);
    let proof = trait_::proof(
        loc, "p", "Tr_tuple", &[(u, i32_ty)], tr_tuple_i32, &[Some("Tr_i32"), None])
        .expect("the proof builds");
    module.body().append_operation(proof);

    // A derive at T with one premise per where-clause entry.
    let tr_t = trait_::trait_application_attr(&context, "Tr", &[t]);
    let tr_t_claim: melior::ir::Type = trait_::claim_type(&context, tr_t).into();
    let out_of_t = trait_::projection_type(&context, tr_t, "Out", &[]);
    let out_is_i64 = trait_::equality_claim_type(&context, out_of_t, i64_ty)
        .expect("the equality claim constructs");
    let block = Block::new(&[(tr_t_claim, loc), (out_is_i64, loc)]);
    let tr_tuple_t = trait_::trait_application_attr(&context, "Tr", &[tuple_of(t)]);
    block.append_operation(
        trait_::derive(
            loc, tr_tuple_t, "Tr_tuple", &[(u, t)],
            &[block.argument(0).unwrap().into(), block.argument(1).unwrap().into()])
            .expect("the derive builds"));
    block.append_operation(func::r#return(&[], loc));
    let body = Region::new();
    body.append_block(block);
    let vis_id = Identifier::new(&context, "sym_visibility");
    let private_attr = StringAttribute::new(&context, "private").into();
    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "g"),
        TypeAttribute::new(FunctionType::new(&context, &[tr_t_claim, out_is_i64], &[]).into()),
        body,
        &[(vis_id, private_attr)],
        loc,
    ));

    assert!(module.as_operation().verify());
    let rendered = module.as_operation().to_string();
    assert!(
        rendered.contains("proves @Tr_tuple[!trait.poly<1> = i32] for @Tr[tuple<i32>] given [@Tr_i32, unit]")
            && rendered.contains("from @Tr_tuple[!trait.poly<1> = !trait.poly<0>] given(%arg0, %arg1)"),
        "the proof and the derive print the arguments they state: {rendered}"
    );
}

#[test]
fn an_impl_instantiates_at_the_arguments_a_derive_states() {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();

    let source = "\
trait.trait private @A[!trait.poly<0>] { trait.assoc_type @Out }\n\
trait.trait private @Tr[!trait.poly<0>] {}\n\
trait.impl private @Tr_tuple for @Tr[tuple<!trait.poly<1>, !trait.poly<0>>] where [@A[!trait.poly<1>], !trait.proj<@A[!trait.poly<1>], \"Out\"> = !trait.poly<0>] {}\n";
    let module = Module::parse(&context, source).expect("the fixture module parses");
    let i32_ty: melior::ir::Type = IntegerType::new(&context, 32).into();
    let i64_ty: melior::ir::Type = IntegerType::new(&context, 64).into();
    let claim = |text: &str| melior::ir::Type::parse(&context, text).expect("the claim parses");

    let arguments = [(trait_::poly_type(&context, 1), i32_ty), (trait_::poly_type(&context, 0), i64_ty)];
    assert_eq!(
        trait_::instantiate_impl(&context, &module, "Tr_tuple", &arguments),
        Ok((
            claim("!trait.claim<@Tr[tuple<i32, i64>]>"),
            vec![claim("!trait.claim<@A[i32]>"), claim("!trait.claim<!trait.proj<@A[i32], \"Out\"> = i64>")],
        )),
    );
    let foreign = [(trait_::poly_type(&context, 5), i32_ty)];
    assert_eq!(
        trait_::instantiate_impl(&context, &module, "Tr_tuple", &foreign),
        Err(trait_::ImplRefusal::NotItsParameters),
    );
    assert_eq!(
        trait_::instantiate_impl(&context, &module, "Tr_missing", &arguments),
        Err(trait_::ImplRefusal::Absent),
    );
    assert_eq!(
        trait_::instantiate_impl(&context, &module, "Tr", &arguments),
        Err(trait_::ImplRefusal::Absent),
        "a trait is no impl"
    );
}
