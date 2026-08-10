// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A claim-respell coerce may respell its operand claim's application but must
// carry the operand's proof forward unchanged: exchanging the backing proof for
// another (here @Safe_proof on the operand for @Safe_proof_alt on the result) is
// a receipt swap the deep no-swap clause refuses, even though both proofs back
// the same application. A respell compares modulo the receipt; it does not
// launder one impl's proof into another's.

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

trait.proof @Safe_proof proves @Safe_impl for @Safe[i32, i64] given []
trait.proof @Safe_proof_alt proves @Safe_impl_alt for @Safe[i32, i64] given []

func.func @swap() -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt> {
  %safe = trait.witness @Safe_proof for @Safe[i32, i64]
  %eq = trait.witness proj_resolve !trait.proj<@Conv[i1], "At"> resolves i64 by @Conv_i1
    : !trait.claim<!trait.proj<@Conv[i1], "At"> = i64>
  // expected-error @below {{may not swap the proof backing claim}}
  %c = trait.coerce %safe : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
    via (%eq) : (!trait.claim<!trait.proj<@Conv[i1], "At"> = i64>)
  return %c : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof_alt>
}
