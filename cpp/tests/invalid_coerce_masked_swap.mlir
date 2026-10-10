// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A coerce compares application claims modulo the label, but may not exchange
// one proof for another. The two endpoints denote one claim once reconciled, so
// naming a different proof on the result than the input carries is a swap the
// verifier refuses. Two impls of one application, each with its own proof, make
// the swap spellable.

trait.trait private @Safe(%self: !trait.claim<@Safe[!trait.poly<0>, !trait.poly<1>]>) {}

trait.impl private @Safe_impl(%self: !trait.claim<@Safe[i32, i64]>) {}
trait.impl private @Safe_impl_alt(%self: !trait.claim<@Safe[i32, i64]>) {}

trait.proof private @Safe_proof {
  %d = trait.derive @Safe[i32, i64] from @Safe_impl given()
  trait.return %d : !trait.claim<@Safe[i32, i64]>
}
trait.proof private @Safe_proof_alt {
  %d = trait.derive @Safe[i32, i64] from @Safe_impl_alt given()
  trait.return %d : !trait.claim<@Safe[i32, i64]>
}

func.func @swap() -> !trait.claim<@Safe[i32, i64] by @Safe_proof_alt> {
  %s = trait.witness @Safe_proof for @Safe[i32, i64]
  // expected-error @below {{may not swap the proof backing claim #trait<application@Safe[i32, i64]>}}
  %c = trait.coerce %s
    : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, i64] by @Safe_proof_alt>
  return %c : !trait.claim<@Safe[i32, i64] by @Safe_proof_alt>
}

// -----

// Wrapping each claim in a container does not hide the swap: a coerce respells
// a claim at its root or a type that holds no claim, so a coerce of a tuple
// holding claims is refused before any proof is compared.

trait.trait private @Safe(%self: !trait.claim<@Safe[!trait.poly<0>, !trait.poly<1>]>) {}

trait.impl private @Safe_impl(%self: !trait.claim<@Safe[i32, i64]>) {}
trait.impl private @Safe_impl_alt(%self: !trait.claim<@Safe[i32, i64]>) {}

trait.proof private @Safe_proof {
  %d = trait.derive @Safe[i32, i64] from @Safe_impl given()
  trait.return %d : !trait.claim<@Safe[i32, i64]>
}
trait.proof private @Safe_proof_alt {
  %d = trait.derive @Safe[i32, i64] from @Safe_impl_alt given()
  trait.return %d : !trait.claim<@Safe[i32, i64]>
}

func.func @masked_swap(%s: tuple<!trait.claim<@Safe[i32, i64] by @Safe_proof>>)
    -> tuple<!trait.claim<@Safe[i32, i64] by @Safe_proof_alt>> {
  // expected-error @below {{may not respell the claim nested in 'tuple<!trait.claim<@Safe[i32, i64] by @Safe_proof>>'}}
  %c = trait.coerce %s
    : tuple<!trait.claim<@Safe[i32, i64] by @Safe_proof>>
    to tuple<!trait.claim<@Safe[i32, i64] by @Safe_proof_alt>>
  return %c : tuple<!trait.claim<@Safe[i32, i64] by @Safe_proof_alt>>
}
