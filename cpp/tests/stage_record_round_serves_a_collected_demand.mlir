// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census served'

// Two impls bind @Other[i64], so the read-only lookup declines and records
// @Other[i64]::X as a demand a round could be asked to serve. Deciding between
// the two is a premise partition, which is impl selection's work and not the
// lookup's: only @Other_wide's assumption holds. So the round that takes the
// demand off the drain answers it.
//
// What each round settled is on its own line, because a demand asked about and
// answered is not the same as one asked about and declined. Counting the ops
// serving inserted cannot tell them apart: selecting an impl that already
// exists writes no IR at all, which is what this row's zero says.
//
// The demand is reached only while the resolver probes candidates for @Gen[i64]
// -- @Gen's one impl spells its own self application through @Other[i64]::X --
// so nothing in the module ever spells it. That is why the stage-exit check
// reports no served drainable key: the key it would report is the one the
// round served on purpose.

!T = !trait.poly<0>

trait.trait @Mark[!T] {}

trait.impl @Mark_i32 for @Mark[i32] {}

trait.trait @Other[!T] {
  trait.assoc_type @X
}

trait.impl @Other_wide for @Other[i64] where [@Mark[i32]] {
  trait.assoc_type @X = i32
}

trait.impl @Other_narrow for @Other[i64] where [@Mark[i16]] {
  trait.assoc_type @X = i16
}

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_via for @Gen[!trait.proj<@Other[i64], "X">] {
  trait.assoc_type @A = i32
}

trait.trait @Box[!T] {}

trait.impl @Box_i32 for @Box[i32] {}

func.func private @f(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i32>,
                     %x: !T) -> !T {
  return %x : !T
}

// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=1
// CHECK-SAME: multiple-candidate-impls=1
// CHECK-SAME: served=1 declined=0 deferred=0 inserted-serving-demands=0
