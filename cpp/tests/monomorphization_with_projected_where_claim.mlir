// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// A trait-level where clause whose obligation carries projections in
// TRAIT-ARGUMENT position (@MyEq[Has::A, Has::A]), discharged inside a
// generic function body by trait.project from the trait claim parameter.
// The projection spelling first becomes resolvable inside the cloned
// callee body, so the call-boundary substitution cannot prove the
// projected claim; proving must happen at the project op itself.
//
// The producer idiom is witness + prebuilt proof (what a proof-carrying
// frontend emits), NOT trait.allege: no allege means the resolver memo is
// not populated transitively, which is what distinguishes this test from
// monomorphization_with_assoc_type_bound.mlir.

// CHECK-NOT: trait.project
// CHECK-NOT: trait.method.call

trait.trait @MyEq[!trait.poly<0>, !trait.poly<1>] {
  func.func nested @eq(!trait.poly<0>, !trait.poly<1>) -> i1
}
trait.trait @Has[!trait.poly<2>] where [@MyEq[!trait.proj<@Has[!trait.poly<2>], "A">, !trait.proj<@Has[!trait.poly<2>], "A">]] {
  trait.assoc_type @A
  func.func nested @get(!trait.poly<2>) -> !trait.proj<@Has[!trait.poly<2>], "A">
}
trait.impl @MyEq_impl for @MyEq[i64, i64] {
  func.func nested @eq(%arg0: i64, %arg1: i64) -> i1 {
    %true = arith.constant true
    return %true : i1
  }
}
trait.impl @Has_impl for @Has[f64] {
  trait.assoc_type @A = i64
  func.func nested @get(%arg0: f64) -> i64 {
    %c0_i64 = arith.constant 0 : i64
    return %c0_i64 : i64
  }
}

// CHECK-LABEL: func.func nested @f_
// CHECK: %[[ID:.*]] = call @Has_impl_get
// CHECK: %[[RES:.*]] = call @MyEq_impl_eq(%[[ID]], %[[ID]])
// CHECK: return %[[RES]]
func.func nested @f(%arg0: !trait.poly<3>, %arg1: !trait.claim<@Has[!trait.poly<3>]>) -> i1 {
  %0 = trait.method.call %arg1 @Has[!trait.poly<3>]::@get(%arg0)
    : (!trait.poly<3>) -> !trait.proj<@Has[!trait.poly<3>], "A">
  %1 = trait.project %arg1: @Has[!trait.poly<3>] to @MyEq[!trait.proj<@Has[!trait.poly<3>], "A">, !trait.proj<@Has[!trait.poly<3>], "A">]
  %2 = trait.method.call %1 @MyEq[!trait.proj<@Has[!trait.poly<3>], "A">, !trait.proj<@Has[!trait.poly<3>], "A">]::@eq(%0, %0)
    : (!trait.proj<@Has[!trait.poly<3>], "A">, !trait.proj<@Has[!trait.poly<3>], "A">) -> i1
  return %2 : i1
}

// CHECK-LABEL: func.func @main
// CHECK: call @f_
func.func @main() -> i32 {
  %cst = arith.constant 0.0 : f64
  %0 = trait.witness @Has_impl_p for @Has[f64]
  %1 = trait.func.call @f(%cst, %0) : (f64, !trait.claim<@Has[f64] by @Has_impl_p>) -> i1
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32 : i32
}
trait.proof @Has_impl_p proves @Has_impl for @Has[f64] given [@MyEq_impl]
