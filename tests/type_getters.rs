// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use melior::{
    Context,
    ir::{r#type::IntegerType, Type},
};
use trait_dialect as trait_;

fn context() -> Context {
    let context = Context::new();
    trait_::register(&context);
    context
}

#[test]
fn a_poly_reads_back_its_label_and_no_other_type_has_one() {
    let context = context();
    assert_eq!(trait_::poly_label(trait_::poly_type(&context, 3)), Some(3));
    assert_eq!(trait_::poly_label(IntegerType::new(&context, 64).into()), None);
}

#[test]
fn an_application_claim_reads_back_its_trait_and_arguments() {
    let context = context();
    let i64_ty: Type = IntegerType::new(&context, 64).into();
    let poly = trait_::poly_type(&context, 0);
    let app = trait_::trait_application_attr(&context, "m.Trait", &[poly, i64_ty]);
    let claim = trait_::claim_type(&context, app);
    let trait_::ClaimPredicate::Trait(read) = claim.predicate() else {
        panic!("an application claim states an application");
    };
    assert_eq!(read.trait_name(), "m.Trait");
    assert_eq!(read.type_args(), vec![poly, i64_ty]);
}

#[test]
fn an_equality_claim_reads_back_its_two_sides() {
    let context = context();
    let i64_ty: Type = IntegerType::new(&context, 64).into();
    let poly = trait_::poly_type(&context, 1);
    let claim = trait_::equality_claim_type(&context, poly, i64_ty).expect("the equality is well formed");
    let claim = trait_::ClaimType::try_from(claim).expect("an equality claim is a claim");
    let trait_::ClaimPredicate::Equal(lhs, rhs) = claim.predicate() else {
        panic!("an equality claim states an equality");
    };
    assert_eq!((lhs, rhs), (poly, i64_ty));
}

#[test]
fn a_projection_reads_back_its_application_name_and_own_arguments() {
    let context = context();
    let i64_ty: Type = IntegerType::new(&context, 64).into();
    let poly = trait_::poly_type(&context, 0);
    let app = trait_::trait_application_attr(&context, "Trait", &[poly]);
    let projection = trait_::projection_type(&context, app, "Assoc", &[i64_ty]);
    let projection = trait_::ProjectionType::try_from(projection).expect("a projection type");
    assert!(projection.trait_application() == app);
    assert_eq!(projection.assoc_name(), "Assoc");
    assert_eq!(projection.assoc_type_args(), vec![i64_ty]);
    assert!(trait_::ProjectionType::try_from(i64_ty).is_err());
}
