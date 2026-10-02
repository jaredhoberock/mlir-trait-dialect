// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T0 = !trait.poly<0>
// CHECK-NOT: trait.trait private @A
trait.trait private @A(%self: !trait.claim<@A[!T0]>) {
  trait.method @method_a(!T0) -> i32
}

!T1 = !trait.poly<1>
// CHECK-NOT: trait.trait private @B
trait.trait private @B(%self: !trait.claim<@B[!T1]>) {
  trait.method @method_b(!T1) -> i32
}

// CHECK-NOT: trait.impl private @B_impl(%self: !trait.claim<@B[i32]>
trait.impl private @B_impl_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @method_b(%arg0: i32) -> i32 {
    %res = arith.constant 1 : i32
    trait.return %res : i32
  }
}

// CHECK-NOT: trait.impl private @B_impl1(%self: !trait.claim<@B[i8]>
trait.impl private @B_impl_i8(%self: !trait.claim<@B[i8]>) {
  trait.method @method_b(%arg: i8) -> i32 {
    %res = arith.constant 1 : i32
    trait.return %res : i32
  }
}

!T2 = !trait.poly<2>
// CHECK-NOT: trait.impl private @A_impl_poly
trait.impl private @A_impl_poly(%self: !trait.claim<@A[!T2]>, %b_1: !trait.claim<@B[!T2]>) {
  trait.method @method_a(%arg0: !T2) -> i32 {
    %res = trait.method.call %b_1 @B[!T2]::@method_b(%arg0)
      : (!T2) -> i32
    trait.return %res : i32
  }
}

// CHECK-LABEL: func.func @test
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @test() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %c7_i8 = arith.constant 7 : i8

  %a_i32 = trait.allege @A[i32]
  // CHECK: call @A_impl_poly_{{.*}}_method_a
  %res0 = trait.method.call %a_i32 @A[i32]::@method_a(%c42_i32)
    : (i32) -> i32

  %a_i8 = trait.allege @A[i8]
  // CHECK: call @A_impl_poly_{{.*}}_method_a
  %res1 = trait.method.call %a_i8 @A[i8]::@method_a(%c7_i8)
    : (i8) -> i32

  %res = arith.addi %res0, %res1 : i32
  return %res : i32
}

// CHECK-NOT: trait.proof
