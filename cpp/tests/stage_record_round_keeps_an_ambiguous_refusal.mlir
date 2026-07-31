// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s

// Two impls bind @Gen[i64] and both their assumption lists are empty, so impl
// selection refuses the application for having two satisfiable candidates. That
// is the refusal no later round overturns -- candidates are only ever appended,
// so a partition that already had two keeps at least two -- which is why the
// next round's flush keeps it where it forgets the other kind, and why the
// demand it refused leaves the drain rather than waiting for facts that cannot
// help it.
//
// @Res[i64]::X is the second demand and the one that keeps the loop running:
// the round that refuses @Gen[i64] serves it, so there is a round after to
// report what the flush kept.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_wide for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.impl @Gen_narrow for @Gen[i64] {
  trait.assoc_type @A = i16
}

trait.trait @Res[!T] {
  trait.assoc_type @X
}

trait.impl @Res_i64 for @Res[i64] {
  trait.assoc_type @X = i32
}

func.func @main() -> (!trait.proj<@Gen[i64], "A">, !trait.proj<@Res[i64], "X">) {
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Gen[i64], "A">
  %s = ub.poison : !trait.proj<@Res[i64], "X">
  return %r, %s : !trait.proj<@Gen[i64], "A">, !trait.proj<@Res[i64], "X">
}

// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=2 no-candidate-impl=0 multiple-candidate-impls=0 other-arms=0 without-arm=2
// CHECK-SAME: served=1 declined=1 deferred=0
// CHECK-SAME: refusals-forgotten=0 refusals-kept=0
// CHECK: trait-stage-record round index=3
// CHECK-SAME: served=0 declined=0 deferred=0
// CHECK-SAME: refusals-forgotten=0 refusals-kept=1 refusals-overturned=0 refusals-re-earned=0
// CHECK: trait-stage-record digest value={{.*}} selected-impls=1 refusals-no-candidate=0 refusals-ambiguous=1
