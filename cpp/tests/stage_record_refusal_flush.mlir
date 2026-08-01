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
// next round's flush drops. @Absent[i64]::B has no impl at all, so the round
// that asks about it refuses, and the flush after drops that refusal too.
//
// Nothing here is re-earned or overturned. A round asks about a demand again
// only where impl selection has minted something since, and selecting impls
// that already exist mints nothing, so each of these refusals is derived once
// and forgotten once.

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
  %r = ub.poison : !trait.proj<@Absent[!T], "B">
  return %r : !trait.proj<@Absent[!T], "B">
}

func.func @main() -> !trait.proj<@Absent[i64], "B"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Absent[i64], "B">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Absent[i64], "B">
  return %r : !trait.proj<@Absent[i64], "B">
}

// Round two collects both demands the driver's read left standing: it answers
// one and refuses the other, with nothing to forget at its own head. The call
// that left @Absent[i64]::B standing put it to the module's impls first, so it
// arrives naming the arm it missed on; @Gen[i64]::A was left standing inside a
// claim the read was normalizing and arrives with none.
// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=2 no-candidate-impl=1 multiple-candidate-impls=0 other-arms=0 without-arm=1
// CHECK-SAME: served=1 declined=1 deferred=1
// CHECK-SAME: refusals-forgotten=0 refusals-kept=0 refusals-overturned=0 refusals-re-earned=0

// Round three forgets what round two refused and answers the demand that
// answering @Gen[i64] raised; round four forgets the refusal that answer left
// behind, and neither round re-derives what its flush dropped.
// CHECK: trait-stage-record round index=3
// CHECK-SAME: collected=1 no-candidate-impl=0 multiple-candidate-impls=1
// CHECK-SAME: served=1 declined=0 deferred=0
// CHECK-SAME: refusals-forgotten=1 refusals-kept=0 refusals-overturned=0 refusals-re-earned=0
// CHECK: trait-stage-record round index=4
// CHECK-SAME: refusals-forgotten=1 refusals-kept=0 refusals-overturned=0 refusals-re-earned=0
