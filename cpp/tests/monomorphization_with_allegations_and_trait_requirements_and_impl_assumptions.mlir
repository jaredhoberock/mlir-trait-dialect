// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK-NOT: trait.trait private @A
trait.trait private @A(%self: !trait.claim<@A[!A]>) {}

!Ai = !trait.poly<1>
// CHECK-NOT: trait.impl private @A_impl
trait.impl private @A_impl(%self: !trait.claim<@A[!Ai]>) {}

!B = !trait.poly<2>
// CHECK-NOT: trait.trait private @B
trait.trait private @B(%self: !trait.claim<@B[!B]>) {}

!Bi = !trait.poly<3>
// CHECK-NOT: trait.impl private @B_impl
trait.impl private @B_impl(%self: !trait.claim<@B[!Bi]>) {}

!C = !trait.poly<4>
// CHECK-NOT: trait.trait private @C
trait.trait private @C(%self_claim: !trait.claim<@C[!C]>) -> !trait.claim<@A[!C]> {
  trait.method @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    trait.return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK-NOT: trait.impl private @C_impl
trait.impl private @C_impl(%self: !trait.claim<@C[!Ci]>, %b: !trait.claim<@B[!Ci]>) {
  %req0 = trait.allege @A[!Ci]
  trait.return %req0 : !trait.claim<@A[!Ci]>
}

// CHECK-LABEL: func.func @foo
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @foo(%x: i8) -> i1 {
  %c = trait.allege @C[i8]
  // CHECK: call @C_{{h[0-9a-f]+}}_method
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// CHECK-NOT: trait.proof
