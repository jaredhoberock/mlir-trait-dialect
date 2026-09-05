// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK-NOT: trait.trait private @A
trait.trait private @A[!A] {}

!Ai = !trait.poly<1>
// CHECK-NOT: trait.impl private @A_impl
trait.impl private @A_impl for @A[!Ai] {}

!B = !trait.poly<2>
// CHECK-NOT: trait.trait private @B
trait.trait private @B[!B] {}

!Bi = !trait.poly<3>
// CHECK-NOT: trait.impl private @B_impl
trait.impl private @B_impl for @B[!Bi] {}

!C = !trait.poly<4>
// CHECK-NOT: trait.trait private @C
trait.trait private @C[!C] where [
  @A[!C]
] {
  func.func @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK-NOT: trait.impl private @C_impl
trait.impl private @C_impl for @C[!Ci] where [
  @B[!Ci]
] {}

// CHECK-LABEL: func.func @foo
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @foo(%x: i8) -> i1 {
  %c = trait.allege @C[i8]
  // CHECK: call @C_impl_{{.*}}_method
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// CHECK-NOT: trait.proof
