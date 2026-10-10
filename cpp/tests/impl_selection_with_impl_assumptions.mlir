// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

!T0 = !trait.poly<0>
// CHECK: trait.trait private @A
trait.trait private @A(%self: !trait.claim<@A[!T0]>) {
  trait.method @method_a(!T0) -> i32
}

!T1 = !trait.poly<1>
// CHECK: trait.trait private @B
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @method_b(!trait.poly<0>) -> i32
}

// CHECK: trait.impl private @B_impl_i32(%self: !trait.claim<@B[i32]>
trait.impl private @B_impl_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @method_b(%arg0: i32) -> i32 {
    %res = arith.constant 1 : i32
    trait.return %res : i32
  }
}

// CHECK: trait.impl private @B_impl_i8(%self: !trait.claim<@B[i8]>
trait.impl private @B_impl_i8(%self: !trait.claim<@B[i8]>) {
  trait.method @method_b(%arg: i8) -> i32 {
    %res = arith.constant 1 : i32
    trait.return %res : i32
  }
}


!T2 = !trait.poly<2>
// CHECK: trait.impl private @A_impl_poly
trait.impl private @A_impl_poly(%self: !trait.claim<@A[!trait.poly<0>]>, %b_1: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @method_a(%arg0: !trait.poly<0>) -> i32 {
    %res = trait.method.call %b_1 @B[!trait.poly<0>]::@method_b(%arg0)
      : (!trait.poly<0>) -> i32
    trait.return %res : i32
  }
}

// CHECK: func.func @test
func.func @test() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %c7_i8 = arith.constant 7 : i8

  // CHECK: trait.witness @A_impl_poly_{{.*}}_p for @A[i32]
  %a_i32 = trait.allege @A[i32]
  %res0 = trait.method.call %a_i32 @A[i32]::@method_a(%c42_i32)
    : (i32) -> i32

  // CHECK: trait.witness @A_impl_poly_{{.*}}_p for @A[i8]
  %a_i8 = trait.allege @A[i8]
  %res1 = trait.method.call %a_i8 @A[i8]::@method_a(%c7_i8)
    : (i8) -> i32

  %res = arith.addi %res0, %res1 : i32
  return %res : i32
}

// CHECK: trait.proof private @A_impl_poly_{{.*}}_p {
// CHECK-NEXT: %[[B8:.*]] = trait.witness @B_impl_i8 for @B[i8]
// CHECK-NEXT: trait.derive @A[i8] from @A_impl_poly given(%[[B8]])
// CHECK: trait.proof private @A_impl_poly_{{.*}}_p {
// CHECK-NEXT: %[[B32:.*]] = trait.witness @B_impl_i32 for @B[i32]
// CHECK-NEXT: trait.derive @A[i32] from @A_impl_poly given(%[[B32]])
