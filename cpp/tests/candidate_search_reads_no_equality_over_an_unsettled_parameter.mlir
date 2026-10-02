// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

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

trait.trait private @Call(%self: !trait.claim<@Call[!S, !A]>) {
  trait.assoc_type @Output
}
trait.trait private @Other(%self: !trait.claim<@Other[!S]>) {
  trait.assoc_type @Out
}
trait.trait private @Pair(%self: !trait.claim<@Pair[!S, !A]>) {
  trait.assoc_type @Res
}

trait.impl private @other_i8(%self: !trait.claim<@Other[i8]>) {
  trait.assoc_type @Out = i32
}
trait.impl private @direct(%self: !trait.claim<@Call[i32, i64]>) {
  trait.assoc_type @Output = f32
}
trait.impl private @chain(%self: !trait.claim<@Pair[!F, !B]>, %output: !trait.claim<!trait.proj<@Call[!G, !B], "Output"> = !Y>, %out: !trait.claim<!trait.proj<@Other[!F], "Out"> = !G>) {
  trait.assoc_type @Res = !Y
}

func.func private @need(!trait.claim<@Pair[i8, i64]>)

// CHECK-LABEL: func.func @main
// CHECK: trait.witness @[[PROOF:.*]] for @Pair[i8, i64]
// CHECK: trait.proof private @[[PROOF]] {
// CHECK: %[[OUTPUT:.*]] = trait.witness proj_resolve !trait.proj<@Call[i32, i64], "Output"> resolves f32 by @direct
// CHECK: %[[OUT:.*]] = trait.witness proj_resolve !trait.proj<@Other[i8], "Out"> resolves i32 by @other_i8
// CHECK: trait.derive @Pair[i8, i64] from @chain given(%[[OUTPUT]], %[[OUT]])
func.func @main() {
  %c = trait.allege @Pair[i8, i64]
  func.call @need(%c) : (!trait.claim<@Pair[i8, i64]>) -> ()
  return
}
