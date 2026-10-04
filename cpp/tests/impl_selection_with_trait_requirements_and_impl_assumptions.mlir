// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

!A = !trait.poly<0>
// CHECK: trait.trait private @
trait.trait private @A(%self: !trait.claim<@A[!A]>) {}

!Ai = !trait.poly<1>
// CHECK: trait.impl private @A_impl
trait.impl private @A_impl(%self: !trait.claim<@A[!Ai]>) {}

!B = !trait.poly<2>
// CHECK: trait.trait private @B
trait.trait private @B(%self: !trait.claim<@B[!B]>) {}

!Bi = !trait.poly<3>
// CHECK: trait.impl private @B_impl
trait.impl private @B_impl(%self: !trait.claim<@B[!Bi]>) {}

!C = !trait.poly<4>
// CHECK: trait.trait private @C
trait.trait private @C(%self_claim: !trait.claim<@C[!C]>) -> !trait.claim<@A[!C]> {
  trait.method @method(%self: !C) -> i1 {
    %res = arith.constant 0 : i1
    trait.return %res : i1
  }
}

!Ci = !trait.poly<5>
// CHECK: trait.impl private @C_impl
trait.impl private @C_impl(%self: !trait.claim<@C[!Ci]>, %b: !trait.claim<@B[!Ci]>) {
  %req0 = trait.allege @A[!Ci]
  trait.return %req0 : !trait.claim<@A[!Ci]>
}

func.func @foo(%x: i8) -> i1 {
  // CHECK: trait.witness @C_impl_{{.*}}_p for @C[i8]
  %c = trait.allege @C[i8]
  %res = trait.method.call %c @C[i8]::@method(%x)
    : (i8) -> i1
  return %res : i1
}

// The proof of @C[i8] supplies @C_impl's where entry @B[i8] by a proof of its
// own; @C's requirement @A[i8] is no premise of it: @C_impl returns that
// evidence itself.
// CHECK-NOT: trait.proof private @A_impl
// CHECK: trait.proof private @B_impl_{{.*}}_p {
// CHECK-NEXT: trait.derive @B[i8] from @B_impl given()
// CHECK: trait.proof private @C_impl_{{.*}}_p {
// CHECK-NEXT: %[[B:.*]] = trait.witness @B_impl_{{.*}}_p for @B[i8]
// CHECK-NEXT: trait.derive @C[i8] from @C_impl given(%[[B]])
