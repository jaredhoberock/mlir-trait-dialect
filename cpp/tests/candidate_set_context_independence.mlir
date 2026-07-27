// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A committed-fact match reads a claim's candidate set the same way in every
// context: it selects by head match alone (premise-blind) and each candidate's
// verdict ignores its peers (peer-blind). @Gen[i64]'s sole candidate here is a
// CONDITIONAL impl whose premise @Needs[i64] no impl proves; a second impl for
// the unrelated application @Gen[f64] is a peer that must never enter
// @Gen[i64]'s candidate set. The witness self-claim match therefore resolves
// @Gen[i64]::A to i32 through the one head-matching candidate, without evaluating
// its premise and without consulting the peer, and matches @T_i32 -- the same
// candidate set resolveGroundProjectionsByLookup reads wherever it runs, so the
// outcome does not depend on whether a verifier or a pass drives it.

trait.trait @Needs[!trait.poly<0>] {}

trait.trait @Gen[!trait.poly<1>] {
  trait.assoc_type @A
}

trait.impl @Gen_cond for @Gen[i64] where [@Needs[i64]] {
  trait.assoc_type @A = i32
}

trait.impl @Gen_peer for @Gen[f64] where [@Needs[f64]] {
  trait.assoc_type @A = f32
}

trait.trait @T[!trait.poly<2>] {}

trait.impl @T_i32 for @T[i32] {}

func.func @f() {
  %w = trait.witness @T_i32 for @T[!trait.proj<@Gen[i64], "A">]
  return
}
