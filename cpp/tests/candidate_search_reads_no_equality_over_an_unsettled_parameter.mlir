// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

// Proving the allegation `@Pair[i8, i64]` selects `@chain`, whose header binds
// `!F` and `!B` and leaves `!G` and `!Y` to its where clause. The first
// equality determines `!Y` from `Call[!G, !B]::Output`, which still names the
// unsettled `!G` in the first round of the reading: it says nothing yet and is
// not normalized, since reading it now would bind `!Y` to a projection over a
// parameter no argument fills. The second equality settles `!G` at `i32`
// through `@other_i8`, and the next round reads the first through `@direct`:
// `!Y` is `f32`. The proof's premises spell both equalities at those
// arguments.

!S = !trait.poly<0>
!A = !trait.poly<1>
!F = !trait.poly<2>
!G = !trait.poly<3>
!Y = !trait.poly<4>
!B = !trait.poly<5>

trait.trait private @Call(!trait.claim<@Call[!S, !A]>) assoc {Output = 0}
trait.trait private @Other(!trait.claim<@Other[!S]>) assoc {Out = 0}
trait.trait private @Pair(!trait.claim<@Pair[!S, !A]>) assoc {Res = 0}

trait.impl private @other_i8(%self: !trait.claim<@Other[i8]>) bindings {Out = i32} {
}
trait.impl private @direct(%self: !trait.claim<@Call[i32, i64]>) bindings {Output = f32} {
}
trait.impl private @chain(%self: !trait.claim<@Pair[!trait.poly<0>, !trait.poly<1>]>, %output: !trait.claim<!trait.proj<@Call[!trait.poly<2>, !trait.poly<1>], "Output"> = !trait.poly<3>>, %out: !trait.claim<!trait.proj<@Other[!trait.poly<0>], "Out"> = !trait.poly<2>>) bindings {Res = !trait.poly<3>} {
}

// A function with a body taking the claim is a template, cloned at the proof
// its call supplies.
func.func private @need(%c: !trait.claim<@Pair[i8, i64]>) {
  return
}

// CHECK-LABEL: func.func @main
// CHECK: trait.witness @[[PROOF:.*]] for @Pair[i8, i64]
// CHECK: trait.proof private @[[PROOF]] {
// CHECK: %[[OUTPUT:.*]] = trait.witness proj_resolve !trait.proj<@Call[i32, i64], "Output"> resolves f32 by @direct
// CHECK: %[[OUT:.*]] = trait.witness proj_resolve !trait.proj<@Other[i8], "Out"> resolves i32 by @other_i8
// CHECK: trait.derive @Pair[i8, i64] from @chain[i8, i64, i32, f32] given(%[[OUTPUT]], %[[OUT]])
func.func @main() {
  %c = trait.allege @Pair[i8, i64]
  trait.func.call @need(%c) : (!trait.claim<@Pair[i8, i64]>) -> ()
  return
}
