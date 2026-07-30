// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census respelling-disagreement'
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=IR

// Respelling a type asks the proof memo about the claims that type spells.
// Building a substitution over the whole memo and applying it is the same
// rewrite, and under the census check the stage performs both and reports every
// position they spell differently.
//
// The two walk a type differently, so this module puts what separates them in
// one place. @take's parameter is a function type spelling a generic type
// beside a claim: a substitution hands the generic type its own specialization
// step and stops the walk there, while the lookup walks the type structurally
// and answers only for the claim. And @Hold's application argument is itself a
// claim, which respelling reaches only by walking into the claim it has just
// proved rather than stopping at it.

!T = !trait.poly<0>

trait.trait @Ground[!T] {}

trait.impl @Ground_all for @Ground[!T] {}

trait.trait @Hold[!T] {
  func.func private @held() -> !T
}

trait.impl @Hold_claim for @Hold[!trait.claim<@Ground[i32]>] where [
  @Ground[i32]
] {
  func.func @held() -> !trait.claim<@Ground[i32]> {
    %g = trait.assume @Ground[i32]
    return %g : !trait.claim<@Ground[i32]>
  }
}

func.func @take(%h: !trait.claim<@Hold[!T]>) -> !T {
  %v = trait.method.call %h @Hold[!T]::@held() : () -> !T
  return %v : !T
}

func.func @test() {
  %h = trait.allege @Hold[!trait.claim<@Ground[i32]>]
  trait.func.call @take(%h)
    : (!trait.claim<@Hold[!trait.claim<@Ground[i32]>]>) -> !trait.claim<@Ground[i32]>
  return
}

// Both claims are proved before the sweep runs, so both are available to
// respell with.
// CHECK: trait-stage-record respelling round=0 bindings=2 ops=6 positions=7

// The module the stage leaves behind spells no claim at all, so nothing was
// left holding a spelling respelling failed to reach.
// IR-NOT: trait.claim
// IR-NOT: builtin.unrealized_conversion_cast
