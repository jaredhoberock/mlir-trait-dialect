// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The swap refusal rides the unproven judgment too. The claim operand is
// unproven, so the verifier reconciles the endpoints by unification rather than
// through an impl -- but the two root claims still name one application under two
// proofs, and a cast may not exchange the proof backing a claim. Both endpoints
// spell the application identically here, so the swap is visible without any
// projection resolving; the masked shapes are pinned separately. This is the
// shape InheritProjCastProofPattern would meet if it ever overwrote an
// already-proven result.

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

func.func @unproven_swap(
    %x: !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof>,
    %conv: !trait.claim<@Conv[i1]>)
    -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt> {
  // expected-error @below {{may not swap the proof backing claim #trait<application@Safe[i32, !trait.proj<@Conv[i1], "At">]>: a cast may drop a proof receipt but not exchange it for another}}
  %cast = trait.proj.cast %x, %conv
    : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
    by !trait.claim<@Conv[i1]>
  return %cast : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
}
