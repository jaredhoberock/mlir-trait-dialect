// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {
  trait.method @a(!trait.poly<0>) -> i1
}
trait.impl private @A_impl_i1(%self: !trait.claim<@A[i1]>) {
  trait.method @a(%arg0: i1) -> i1 {
    trait.return %arg0 : i1
  }
}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @b(%arg0: !trait.poly<0>) -> i1
}
trait.impl private @B_impl_i1(%self: !trait.claim<@B[i1]>) {
  trait.method @b(%arg0: i1) -> i1 {
    %a = trait.allege @A[i1]

    // test that we are able to trait.method.call
    // from inside a method to another trait
    %res = trait.method.call %a @A[i1]::@a(%arg0) : (i1) -> i1
    trait.return %res : i1
  }
}
func.func @test(%arg0 : i1) -> i1 {
  %b = trait.allege @B[i1]
  %res = trait.method.call %b @B[i1]::@b(%arg0) : (i1) -> i1
  return %res : i1
}

// Instance of A::a
// CHECK: func.func private @A_impl_i1_{{h[0-9a-f]+}}_a(

// Instance of B::b that calls A_impl_i1_a
// CHECK: func.func private @B_impl_i1_{{h[0-9a-f]+}}_b(
// CHECK: call @A_impl_i1_{{h[0-9a-f]+}}_a

// Top-level test calls B_impl_i1_b
// CHECK: func.func @test(
// CHECK: call @B_impl_i1_{{h[0-9a-f]+}}_b

// No trait ops should remain
// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.func.call
// CHECK-NOT: trait.allege
