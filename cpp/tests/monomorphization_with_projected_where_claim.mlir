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
// CHECK-NOT: trait.func.call

trait.trait private @MyEq(!trait.claim<@MyEq[!trait.poly<0>, !trait.poly<1>]>) members [@MyEq.eq]
func.func private @MyEq.eq(!trait.claim<@MyEq[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> i1
trait.trait private @Has(!trait.claim<@Has[!trait.poly<0>]>) -> !trait.claim<@MyEq[!trait.proj<@Has[!trait.poly<0>], "A">, !trait.proj<@Has[!trait.poly<0>], "A">]> assoc {A = 0} members [@Has.get]
func.func private @Has.get(!trait.claim<@Has[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Has[!trait.poly<0>], "A">
trait.impl private @MyEq_impl(%self: !trait.claim<@MyEq[i64, i64]>) members [@MyEq.eq -> @MyEq_impl.eq] {
}
func.func private @MyEq_impl.eq(%self: !trait.claim<@MyEq[i64, i64]>, %arg0: i64, %arg1: i64) -> i1 {
  %true = arith.constant true
  return %true : i1
}
trait.impl private @Has_impl(%self: !trait.claim<@Has[f64]>) bindings {A = i64} members [@Has.get -> @Has_impl.get] {
  %req0 = trait.allege @MyEq[!trait.proj<@Has[f64], "A">, !trait.proj<@Has[f64], "A">]
  trait.return %req0 : !trait.claim<@MyEq[!trait.proj<@Has[f64], "A">, !trait.proj<@Has[f64], "A">]>
}
func.func private @Has_impl.get(%self: !trait.claim<@Has[f64]>, %arg0: f64) -> i64 {
  %c0_i64 = arith.constant 0 : i64
  return %c0_i64 : i64
}

// CHECK-LABEL: func.func nested @f_
// CHECK: %[[ID:.*]] = call @Has_impl.get_{{h[0-9a-f]+}}
// CHECK: %[[RES:.*]] = call @MyEq_impl.eq_{{h[0-9a-f]+}}(%[[ID]], %[[ID]])
// CHECK: return %[[RES]]
func.func nested @f(%arg0: !trait.poly<0>, %arg1: !trait.claim<@Has[!trait.poly<0>]>) -> i1 {
  %0 = trait.func.call @Has.get(%arg1, %arg0) : (!trait.claim<@Has[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Has[!trait.poly<0>], "A">
  %1 = trait.project %arg1[0] : !trait.claim<@Has[!trait.poly<0>]> -> !trait.claim<@MyEq[!trait.proj<@Has[!trait.poly<0>], "A">, !trait.proj<@Has[!trait.poly<0>], "A">]>
  %2 = trait.func.call @MyEq.eq(%1, %0, %0) : (!trait.claim<@MyEq[!trait.proj<@Has[!trait.poly<0>], "A">, !trait.proj<@Has[!trait.poly<0>], "A">]>, !trait.proj<@Has[!trait.poly<0>], "A">, !trait.proj<@Has[!trait.poly<0>], "A">) -> i1
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
trait.proof private @Has_impl_p {
  %d = trait.derive @Has[f64] from @Has_impl given()
  trait.return %d : !trait.claim<@Has[f64]>
}
