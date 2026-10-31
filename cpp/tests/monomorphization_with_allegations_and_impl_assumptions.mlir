// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!T0 = !trait.poly<0>
// CHECK-NOT: trait.trait private @A
trait.trait private @A(!trait.claim<@A[!T0]>) members [@A.method_a]
func.func private @A.method_a(!trait.claim<@A[!T0]>, !T0) -> i32

!T1 = !trait.poly<1>
// CHECK-NOT: trait.trait private @B
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) members [@B.method_b]
func.func private @B.method_b(!trait.claim<@B[!trait.poly<0>]>, !trait.poly<0>) -> i32

// CHECK-NOT: trait.impl private @B_impl(%self: !trait.claim<@B[i32]>
trait.impl private @B_impl_i32(%self: !trait.claim<@B[i32]>) members [@B.method_b -> @B_impl_i32.method_b] {
}
func.func private @B_impl_i32.method_b(%self: !trait.claim<@B[i32]>, %arg0: i32) -> i32 {
  %res = arith.constant 1 : i32
  return %res : i32
}

// CHECK-NOT: trait.impl private @B_impl1(%self: !trait.claim<@B[i8]>
trait.impl private @B_impl_i8(%self: !trait.claim<@B[i8]>) members [@B.method_b -> @B_impl_i8.method_b] {
}
func.func private @B_impl_i8.method_b(%self: !trait.claim<@B[i8]>, %arg: i8) -> i32 {
  %res = arith.constant 1 : i32
  return %res : i32
}

!T2 = !trait.poly<2>
// CHECK-NOT: trait.impl private @A_impl_poly
trait.impl private @A_impl_poly(%self: !trait.claim<@A[!trait.poly<0>]>, %b_1: !trait.claim<@B[!trait.poly<0>]>) members [@A.method_a -> @A_impl_poly.method_a] {
}
func.func private @A_impl_poly.method_a(%self: !trait.claim<@A[!trait.poly<0>]>, %b_1: !trait.claim<@B[!trait.poly<0>]>, %arg0: !trait.poly<0>) -> i32 {
  %res = trait.func.call @B.method_b(%b_1, %arg0) : (!trait.claim<@B[!trait.poly<0>]>, !trait.poly<0>) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @test
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @test() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %c7_i8 = arith.constant 7 : i8

  %a_i32 = trait.allege @A[i32]
  // CHECK: call @A_impl_poly.method_a_{{.*}}
  %res0 = trait.func.call @A.method_a(%a_i32, %c42_i32) : (!trait.claim<@A[i32]>, i32) -> i32

  %a_i8 = trait.allege @A[i8]
  // CHECK: call @A_impl_poly.method_a_{{.*}}
  %res1 = trait.func.call @A.method_a(%a_i8, %c7_i8) : (!trait.claim<@A[i8]>, i8) -> i32

  %res = arith.addi %res0, %res1 : i32
  return %res : i32
}

// CHECK-NOT: trait.proof
