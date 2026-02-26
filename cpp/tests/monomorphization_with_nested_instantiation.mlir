// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Verifies that proofs created during the greedy rewrite are propagated
// into claim types of functions instantiated in the same rewrite pass.
//
// @outer derives a claim for @Tr[!G] and passes it to @inner via
// trait.func.call.  When @outer is instantiated with i32, the derive fires
// during the greedy rewrite, but @inner's block argument still holds an
// unproven claim type until the post-rewrite substitution propagates it.

// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.derive
// CHECK-NOT: trait.func.call
// CHECK-NOT: trait.method.call

!T0 = !trait.poly<0>

trait.trait @Tr [!T0] {
  func.func private @method(!T0) -> i32
}

trait.impl @Tr_i32 for @Tr[i32] {
  func.func @method(%self: i32) -> i32 {
    return %self : i32
  }
}

// Takes a claim, calls method through it
!F = !trait.poly<2>
func.func @inner(%x: !F, %c: !trait.claim<@Tr[!F]>) -> i32 {
  %r = trait.method.call %c @Tr[!F]::@method(%x) : (!F) -> i32
  return %r : i32
}

// Derives Tr[!G] (unconditional impl), passes claim to @inner
!G = !trait.poly<3>
func.func @outer(%x: !G) -> i32 {
  %c = trait.derive @Tr[!G] from @Tr_i32 given()
  %r = trait.func.call @inner(%x, %c)
    : (!G, !trait.claim<@Tr[!G]>) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @test
func.func @test(%x: i32) -> i32 {
  %r = trait.func.call @outer(%x) : (i32) -> i32
  return %r : i32
}
