// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{
        BlockLike, Module, Type,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::FunctionType,
    },
    utility::register_all_dialects,
};
use trait_dialect as trait_;

fn context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    trait_::register(&context);
    context.load_all_available_dialects();
    context
}

/// The trait name each of `ty`'s parts applies, for a claim, a projection or
/// an equality claim between two projections.
fn trait_names(ty: Type) -> Vec<String> {
    if let Ok(projection) = trait_::ProjectionType::try_from(ty) {
        return vec![projection.trait_application().trait_name()];
    }
    match trait_::ClaimType::try_from(ty).expect("a claim or a projection").predicate() {
        trait_::ClaimPredicate::Trait(app) => vec![app.trait_name()],
        trait_::ClaimPredicate::Equal(lhs, rhs) => trait_names(lhs).into_iter().chain(trait_names(rhs)).collect(),
    }
}

#[test]
fn a_rename_reaches_an_equality_claims_endpoints_a_citations_arguments_and_every_other_reference() {
    let context = context();
    let text = r#"
    module {
      trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {
        trait.assoc_type @A
        trait.method @m(!trait.poly<0>) -> i64
      }
      func.func private @m(%x: i64) -> i64 {
        return %x : i64
      }
      func.func private @h(%c: !trait.claim<@T[i64]>, %x: i64) -> i64 {
        %y = trait.method.call %c @T[i64]::@m(%x) : (i64) -> i64
        return %y : i64
      }
      trait.impl private @I(%self: !trait.claim<@T[i64]>) {
        trait.assoc_type @A = i64
        trait.method @m(%x: i64) -> i64 {
          trait.return %x : i64
        }
      }
      trait.impl private @J(%self: !trait.claim<@T[tuple<!trait.poly<0>>]>) {
        trait.assoc_type @A = i1
        trait.method @m(%x: tuple<!trait.poly<0>>) -> i64 {
          %one = arith.constant 1 : i64
          trait.return %one : i64
        }
      }
      func.func private @g() -> !trait.claim<@T[tuple<!trait.proj<@T[i64], "A">>]> {
        %c = trait.derive @T[tuple<!trait.proj<@T[i64], "A">>] from @J[!trait.proj<@T[i64], "A">] given()
        return %c : !trait.claim<@T[tuple<!trait.proj<@T[i64], "A">>]>
      }
      func.func private @f(%a: !trait.proj<@T[i64], "A">,
                           %b: !trait.claim<@T[i64]>,
                           %c: !trait.claim<!trait.proj<@T[i64], "A"> = !trait.proj<@T[i1], "A">>) {
        return
      }
    }
    "#;
    let module = Module::parse(&context, text).expect("the module parses");
    // `m` names a top-level function and a method of `T`: the reference
    // `@T::@m` renames its root alone
    let renames = [("T".to_string(), "m.T".to_string()), ("m".to_string(), "m.m".to_string())];
    unsafe { trait_::rename_symbols(module.as_operation().to_raw(), &renames) };
    // the citation's arguments name the renamed trait; it verifies only once
    // they move with the claim it derives
    assert!(module.as_operation().verify());

    let declared = module.body().first_operation().expect("the trait");
    assert_eq!(StringAttribute::try_from(declared.attribute("sym_name").unwrap()).unwrap().value(), "m.T");
    let f = declared
        .next_in_block()
        .and_then(|m| m.next_in_block())
        .and_then(|h| h.next_in_block())
        .and_then(|i| i.next_in_block())
        .and_then(|j| j.next_in_block())
        .and_then(|g| g.next_in_block())
        .expect("the function");
    let fn_ty = FunctionType::try_from(TypeAttribute::try_from(f.attribute("function_type").unwrap()).unwrap().value()).unwrap();
    let names: Vec<String> = (0..fn_ty.input_count()).flat_map(|k| trait_names(fn_ty.input(k).unwrap())).collect();
    assert_eq!(names, ["m.T", "m.T", "m.T", "m.T"]);
}
