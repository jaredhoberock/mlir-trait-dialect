// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A pair the record holds is still the pair a later ask arrives with, across the
// commit sweep that respells the module in between.
//
// The sweep gives an unproven claim its proof, so applying its rewrite to a
// spelling rewrites both what is nested inside that spelling and the spelling
// itself. Transcribing the record with it must take only the first: every
// position the record holds is an obligation or the claim proving one, and
// which it is decides how the next reader spells its ask. An obligation
// transcribed into the proven claim it is paired with is a key nothing asks.
//
// @user1's call to @g asks the @P[i32] pair in round one and the record holds
// what deriving it produced. @user2's call to @h clones @h in round one; the
// clone's derive of @Q[i32] is served in round two and respelled by round two's
// sweep, which is what transcribes the record. The clone's inner call to @g2
// then asks the same @P[i32] pair again, and one new pair for @Q[i32].
//
// So two distinct pairs are asked and exactly one ask of each goes unanswered.
// A transcription that moved a position's grade would leave the @P pair
// unanswered twice.

trait.trait @P[!trait.poly<0>] {}
trait.impl @P_impl for @P[!trait.poly<1>] {}

trait.trait @Q[!trait.poly<2>] {}
trait.impl @Q_impl for @Q[!trait.poly<3>] {}

!T = !trait.poly<4>
func.func @g(%c: !trait.claim<@P[!T]>, %x: !T) -> !T {
  return %x : !T
}

!U = !trait.poly<5>
func.func @g2(%q: !trait.claim<@Q[!U]>, %c: !trait.claim<@P[!U]>, %x: !U) -> !U {
  return %x : !U
}

!V = !trait.poly<6>
func.func @h(%c: !trait.claim<@P[!V]>, %x: !V) -> !V {
  %q = trait.derive @Q[!V] from @Q_impl given()
  %r = trait.func.call @g2(%q, %c, %x)
    : (!trait.claim<@Q[!V]>, !trait.claim<@P[!V]>, !V) -> !V
  return %r : !V
}

func.func @user1(%x: i32) -> i32 {
  %p = trait.allege @P[i32]
  %r = trait.func.call @g(%p, %x) : (!trait.claim<@P[i32]>, i32) -> i32
  return %r : i32
}

func.func @user2(%x: i32) -> i32 {
  %p = trait.allege @P[i32]
  %r = trait.func.call @h(%p, %x) : (!trait.claim<@P[i32]>, i32) -> i32
  return %r : i32
}

// The sweep that transcribes the record is round two's, and it respells: a
// sweep that moved no position does not transcribe, so the row would say
// nothing about transcription without this.
// CHECK: trait-stage-record respelling round=2
// CHECK-SAME: positions=1

// CHECK: trait-demand-census counter proof
// CHECK-SAME: derivations-not-recorded=0
// CHECK-SAME: closures-unanswered=2
// CHECK-SAME: closures-withdrawn=0
