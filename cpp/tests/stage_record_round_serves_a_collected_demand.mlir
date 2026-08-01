// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census served'

// A round collects what the module spells, and @f's declared claim spells
// @Gen[i64]::A in @f's own block argument, so the first round takes that demand
// off the module walk. It arrives naming no arm: the walk reaches the spelling
// before anything has asked the lookup about it, and naming which way an
// application missed is impl selection's to do. Selection answers it in that
// round, and its candidate probe for @Gen[i64] reaches @Other[i64]::X --
// @Gen's one impl spells its own self application through it. Two impls bind
// that, so the read-only lookup declines there on the multiple-candidate arm
// and a second demand is recorded, this one carrying the arm its asking named.
// Deciding between the two is a premise partition, which is impl selection's
// work and not the lookup's: only @Other_wide's assumption holds. So the round
// after takes it off the drain and answers it.
//
// What each round settled is on its own line, because a demand asked about and
// answered is not the same as one asked about and declined. Counting the ops
// serving inserted cannot tell them apart: selecting an impl that already
// exists writes no IR at all, which is what this row's zeros say.
//
// @Other[i64]::X is reached only inside that probe, so nothing in the module
// ever spells it. The stage-exit check names no key all the same: a key the
// rounds served is one whose spelling is gone by design, and this one carries
// the arm the lookup declined it on besides.

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

// CHECK: trait-stage-record round index=1
// CHECK-SAME: collected=1 no-candidate-impl=0 multiple-candidate-impls=0 other-arms=0 without-arm=1
// CHECK-SAME: served=1 declined=0 deferred=0 inserted-serving-demands=0
// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=1
// CHECK-SAME: multiple-candidate-impls=1
// CHECK-SAME: served=1 declined=0 deferred=0 inserted-serving-demands=0
