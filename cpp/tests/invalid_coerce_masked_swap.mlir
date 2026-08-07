// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A coerce compares application receipts modulo the label, but may not exchange
// one proof for another. The two endpoints denote one claim once reconciled, so
// naming a different proof on the result than the input carries is a swap the
// verifier refuses. Two impls of one application, each with its own proof, make
// the swap spellable.

trait.trait @Safe[!trait.poly<0>, !trait.poly<1>] {}

trait.impl @Safe_impl for @Safe[i32, i64] {}
trait.impl @Safe_impl_alt for @Safe[i32, i64] {}

trait.proof @Safe_proof proves @Safe_impl for @Safe[i32, i64] given []
trait.proof @Safe_proof_alt proves @Safe_impl_alt for @Safe[i32, i64] given []

func.func @swap() -> !trait.claim<@Safe[i32, i64] by @Safe_proof_alt> {
  %s = trait.witness @Safe_proof for @Safe[i32, i64]
  // expected-error @below {{may not swap the proof backing claim #trait<application@Safe[i32, i64]>}}
  %c = trait.coerce %s
    : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, i64] by @Safe_proof_alt>
  return %c : !trait.claim<@Safe[i32, i64] by @Safe_proof_alt>
}
