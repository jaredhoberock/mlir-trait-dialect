// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A projection spelling must not mask a proof swap. The input names
// @Safe[i32, i64] proven by @Safe_proof; the result spells the same claim
// through @Conv[i1]::At -- which resolves to i64 -- but names @Safe_proof_alt.
// The claim operand is unproven, so the endpoints are reconciled by
// unification; once they are known to denote one claim, the differing receipt is
// a swap the cast may not perform. The refusal must read the endpoints after
// they reconcile, not the raw spellings that differ by the projection, or the
// swap slips through.

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

func.func @masked_swap(%conv: !trait.claim<@Conv[i1]>)
    -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt> {
  %safe = trait.witness @Safe_proof for @Safe[i32, i64]
  // expected-error @below {{may not swap the proof backing claim #trait<application@Safe[i32, !trait.proj<@Conv[i1], "At">]>: a cast may drop a proof receipt but not exchange it for another}}
  %cast = trait.proj.cast %safe, %conv
    : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
    by !trait.claim<@Conv[i1]>
  return %cast : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
}
