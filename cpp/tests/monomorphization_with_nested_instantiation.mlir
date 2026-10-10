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

trait.trait private @Tr(%self: !trait.claim<@Tr[!T0]>) {
  trait.method @method(!T0) -> i32
}

// A blanket impl: @outer derives its claim at its own type variable, so the
// impl's header must be spelled over one too -- an impl for a single concrete
// type justifies nothing about a variable.
!B = !trait.poly<9>
trait.impl private @Tr_any(%self_claim: !trait.claim<@Tr[!trait.poly<0>]>) {
  trait.method @method(%self: !trait.poly<0>) -> i32 {
    %c = arith.constant 0 : i32
    trait.return %c : i32
  }
}

// Takes a claim, calls method through it
!F = !trait.poly<2>
func.func private @inner(%x: !trait.poly<0>, %c: !trait.claim<@Tr[!trait.poly<0>]>) -> i32 {
  %r = trait.method.call %c @Tr[!trait.poly<0>]::@method(%x) : (!trait.poly<0>) -> i32
  return %r : i32
}

// Derives Tr[!G] (unconditional impl), passes claim to @inner
!G = !trait.poly<3>
func.func private @outer(%x: !trait.poly<0>) -> i32 {
  %c = trait.derive @Tr[!trait.poly<0>] from @Tr_any[!trait.poly<0>] given()
  %r = trait.func.call @inner(%x, %c)
    : (!trait.poly<0>, !trait.claim<@Tr[!trait.poly<0>]>) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @test
func.func @test(%x: i32) -> i32 {
  %r = trait.func.call @outer(%x) : (i32) -> i32
  return %r : i32
}
