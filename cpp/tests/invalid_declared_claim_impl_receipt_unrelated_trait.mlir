// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' -verify-diagnostics

// A `by @impl` receipt naming a bare unconditional impl is audited the same way
// a `by @proof` receipt is: the impl's own self claim must specialize to the
// claim the receipt annotates. Here the signature claims @Other[...] but names
// @HasPart_i64, an impl of an entirely different trait. Naming an unconditional
// impl is not proving the claim, so the mismatch must be refused where the
// receipt is declared rather than trusted through to a leaf binding.

trait.trait @HasPart[!trait.poly<0>] {
  trait.assoc_type @Part
}
trait.impl @HasPart_i64 for @HasPart[i64] {
  trait.assoc_type @Part = f32
}
trait.trait @Other[!trait.poly<0>] {}

// expected-error @below {{declared claim in signature has an invalid proof: trait mismatch: expected @HasPart, but found @Other}}
func.func private @takes_forged(
    %op: !trait.claim<@Other[!trait.proj<@HasPart[i64], "Part">] by @HasPart_i64>,
    %x: !trait.poly<7>
) -> !trait.poly<7> {
  return %x : !trait.poly<7>
}
