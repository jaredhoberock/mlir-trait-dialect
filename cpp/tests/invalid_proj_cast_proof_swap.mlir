// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Comparison modulo the receipt would equate a claim proven by one impl with the
// same claim proven by another -- but a cast may not rewrite which impl proves a
// claim. The result's receipt must be absent or name the same proof as the
// input's; naming a different proof is a swap the verifier refuses. Two impls of
// one application, each with its own proof, make the swap spellable.

!S = !trait.poly<0>

trait.trait @Conv[!S] {
  trait.assoc_type @At
}

trait.trait @Safe[!trait.poly<0>, !trait.poly<1>] {
}

trait.impl @Conv_i1 for @Conv[i1] {
  trait.assoc_type @At = i64
}

trait.impl @Safe_impl for @Safe[i32, i64] {
}

trait.impl @Safe_impl_alt for @Safe[i32, i64] {
}

trait.proof @Conv_proof proves @Conv_i1 for @Conv[i1] given []
trait.proof @Safe_proof proves @Safe_impl for @Safe[i32, i64] given []
trait.proof @Safe_proof_alt proves @Safe_impl_alt for @Safe[i32, i64] given []

func.func @swap() -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt> {
  %safe = trait.witness @Safe_proof for @Safe[i32, i64]
  %conv = trait.witness @Conv_proof for @Conv[i1]
  // expected-error @below {{may not swap the proof backing claim #trait<application@Safe[i32, i64]>: a cast may drop a proof receipt but not exchange it for another}}
  %cast = trait.proj.cast %safe, %conv
    : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
    by !trait.claim<@Conv[i1] by @Conv_proof>
  return %cast : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
}
