// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s

// A refusal for want of a satisfiable candidate is one an impl generated since
// can overturn, so each round forgets those before it asks anything. Whether
// that pays is what the two counts beside the flush say: a forgotten refusal
// the round then re-derived was re-earned, and one it answered with an impl was
// overturned.
//
// This module has one demand of each kind. @Other[i64]::X is refused nowhere --
// two impls bind it and one assumption holds, so the round that collects it
// answers it, and answering leaves a refusal behind for @Mark[i16], which the
// next round's flush drops. @Absent[i64]::B has no impl at all, so every round
// that asks about it refuses again, and the round after sees its own forgotten
// refusal come back.

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

trait.trait @Absent[!T] {
  trait.assoc_type @B
}

func.func @wrap(%x: !T) -> !trait.proj<@Absent[!T], "B"> {
  // expected-error @below {{unresolved projection '!trait.proj<@Absent[i64], "B">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Absent[!T], "B">
  return %r : !trait.proj<@Absent[!T], "B">
}

func.func @main() -> !trait.proj<@Absent[i64], "B"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Absent[i64], "B">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Absent[i64], "B">
  return %r : !trait.proj<@Absent[i64], "B">
}

// Round two collects both demands: it answers one and leaves the other for a
// round the facts have moved under, and the refusal it forgot at its head was
// the one round one recorded.
// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=2 no-candidate-impl=1 multiple-candidate-impls=1
// CHECK-SAME: served=1 declined=1 deferred=1
// CHECK-SAME: refusals-forgotten=1 refusals-kept=0 refusals-overturned=0 refusals-re-earned=0

// Round three forgets what round two refused, and reports that one of the two
// refusals its own flush dropped a round earlier came straight back.
// CHECK: trait-stage-record round index=3
// CHECK-SAME: refusals-forgotten=2 refusals-kept=0 refusals-overturned=0 refusals-re-earned=1
