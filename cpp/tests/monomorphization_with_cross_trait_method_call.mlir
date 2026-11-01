// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s --implicit-check-not=trait.

// No trait op remains anywhere in the output: the implicit check reads every
// line, where a CHECK-NOT after the last match reads only the lines below it.

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>, !trait.poly<0>) -> i1
trait.impl private @A_impl_i1(%self: !trait.claim<@A[i1]>) members [@A.a -> @A_impl_i1.a] {
}
func.func private @A_impl_i1.a(%self: !trait.claim<@A[i1]>, %arg0: i1) -> i1 {
  return %arg0 : i1
}
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!trait.poly<0>]>, !trait.poly<0>) -> i1
trait.impl private @B_impl_i1(%self: !trait.claim<@B[i1]>) members [@B.b -> @B_impl_i1.b] {
}
func.func private @B_impl_i1.b(%self: !trait.claim<@B[i1]>, %arg0: i1) -> i1 {
  %a = trait.allege @A[i1]

  // test that we are able to trait.func.call
  // from inside a member to another trait
  %res = trait.func.call @A.a(%a, %arg0) : (!trait.claim<@A[i1]>, i1) -> i1
  return %res : i1
}
func.func @test(%arg0 : i1) -> i1 {
  %b = trait.allege @B[i1]
  %res = trait.func.call @B.b(%b, %arg0) : (!trait.claim<@B[i1]>, i1) -> i1
  return %res : i1
}

// Instance of A::a
// CHECK: func.func private @A_impl_i1.a_{{h[0-9a-f]+}}(

// Instance of B::b that calls A_impl_i1_a
// CHECK: func.func private @B_impl_i1.b_{{h[0-9a-f]+}}(
// CHECK: call @A_impl_i1.a_{{h[0-9a-f]+}}

// Top-level test calls B_impl_i1_b
// CHECK: func.func @test(
// CHECK: call @B_impl_i1.b_{{h[0-9a-f]+}}
