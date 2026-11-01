// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK-NOT: trait.trait private @A
trait.trait private @A(!trait.claim<@A[!A]>)

!Ai = !trait.poly<1>
// CHECK-NOT: trait.impl private @A_impl
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) {
}

!B = !trait.poly<2>
// CHECK-NOT: trait.trait private @B
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>)

!Bi = !trait.poly<3>
// CHECK-NOT: trait.impl private @B_impl
trait.impl private @B_impl(%self: !trait.claim<@B[!trait.poly<0>]>) {
}

!C = !trait.poly<4>
// CHECK-NOT: trait.trait private @C
trait.trait private @C(!trait.claim<@C[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> members [default @C.method]
func.func private @C.method(%self_claim: !trait.claim<@C[!trait.poly<0>]>, %self: !trait.poly<0>) -> i1 {
  %res = arith.constant 0 : i1
  return %res : i1
}

!Ci = !trait.poly<5>
// CHECK-NOT: trait.impl private @C_impl
trait.impl private @C_impl(%self: !trait.claim<@C[!trait.poly<0>]>, %b: !trait.claim<@B[!trait.poly<0>]>) {
  %req0 = trait.allege @A[!trait.poly<0>]
  trait.return %req0 : !trait.claim<@A[!trait.poly<0>]>
}

// CHECK-LABEL: func.func @foo
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @foo(%x: i8) -> i1 {
  %c = trait.allege @C[i8]
  // CHECK: call @C.method_{{h[0-9a-f]+}}
  %res = trait.func.call @C.method(%c, %x) : (!trait.claim<@C[i8]>, i8) -> i1
  return %res : i1
}

// CHECK-NOT: trait.proof
