// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The record answers a second ask of one pair, and nothing but a sweep can make
// it stop.
//
// This is the companion of the row where a sweep runs in between: the same pair
// asked in two rounds, but round two's serve settles a projection rather than
// proving a claim, so nothing is respelled and the record is never transcribed.
// Whatever the transcription does, it cannot be what answers here -- so a row
// that stops answering with no sweep between the asks has lost the record for
// some other reason, which is what this pins.
//
// @user1 asks the @P[i32] pair in round one. @user2's call to @h is turned away
// in round one over the projection @g2's signature spells; round two selects the
// impl that resolves it, the call lowers, and it asks the same @P[i32] pair.

trait.trait @P[!trait.poly<0>] {}
trait.impl @P_impl for @P[!trait.poly<1>] {}

trait.trait @R[!trait.poly<2>] {
  trait.assoc_type @A
}
trait.impl @R_impl for @R[i32] {
  trait.assoc_type @A = i32
}

!T = !trait.poly<4>
func.func @g(%c: !trait.claim<@P[!T]>, %x: !T) -> !T {
  return %x : !T
}

!U = !trait.poly<5>
func.func @g2(%c: !trait.claim<@P[!U]>, %x: !U) -> !trait.proj<@R[!U], "A"> {
  %r = builtin.unrealized_conversion_cast %x : !U to !trait.proj<@R[!U], "A">
  return %r : !trait.proj<@R[!U], "A">
}

!V = !trait.poly<6>
func.func @h(%c: !trait.claim<@P[!V]>, %x: !V) -> !trait.proj<@R[!V], "A"> {
  %r = trait.func.call @g2(%c, %x)
    : (!trait.claim<@P[!V]>, !V) -> !trait.proj<@R[!V], "A">
  return %r : !trait.proj<@R[!V], "A">
}

func.func @user1(%x: i32) -> i32 {
  %p = trait.allege @P[i32]
  %r = trait.func.call @g(%p, %x) : (!trait.claim<@P[i32]>, i32) -> i32
  return %r : i32
}

func.func @user2(%x: i32) -> i32 {
  %p = trait.allege @P[i32]
  %r = trait.func.call @h(%p, %x)
    : (!trait.claim<@P[i32]>, i32) -> !trait.proj<@R[i32], "A">
  %y = builtin.unrealized_conversion_cast %r : !trait.proj<@R[i32], "A"> to i32
  return %y : i32
}

// One pair, asked in two rounds, unanswered once.
// CHECK: trait-demand-census counter proof
// CHECK-SAME: derivations-not-recorded=0
// CHECK-SAME: closures-unanswered=1
// CHECK-SAME: closures-withdrawn=0
