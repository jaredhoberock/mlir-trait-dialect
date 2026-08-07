// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The same masking must not let a cast attach a proof the input never carried.
// The input names @Safe[i32, i64] and carries no receipt; the result spells the
// same claim through @Conv[i1]::At -- which resolves to i64 -- and names
// @Safe_proof_alt. The claim operand is unproven, so the endpoints are
// reconciled by unification, and a result receipt over an unproven input is an
// attach the cast may not perform. As with the swap, the refusal must read the
// reconciled endpoints rather than the raw projection spelling.

!S = !trait.poly<0>

trait.trait @Conv[!S] {
  trait.assoc_type @At
}

trait.trait @Safe[!trait.poly<0>, !trait.poly<1>] {
}

trait.impl @Conv_i1 for @Conv[i1] {
  trait.assoc_type @At = i64
}

trait.impl @Safe_impl_alt for @Safe[i32, i64] {
}

trait.proof @Conv_proof proves @Conv_i1 for @Conv[i1] given []
trait.proof @Safe_proof_alt proves @Safe_impl_alt for @Safe[i32, i64] given []

func.func @masked_attach(
    %safe: !trait.claim<@Safe[i32, i64]>,
    %conv: !trait.claim<@Conv[i1]>)
    -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt> {
  // expected-error @below {{may not swap the proof backing claim #trait<application@Safe[i32, !trait.proj<@Conv[i1], "At">]>: a cast may drop a proof receipt but not exchange it for another}}
  %cast = trait.proj.cast %safe, %conv
    : !trait.claim<@Safe[i32, i64]>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
    by !trait.claim<@Conv[i1]>
  return %cast : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
}
