// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A premise is compared with the cited impl's where entry as written: the
// entry is @X[i32] at the citation and the witness supplies
// @X[!Other[i32]::A]. @f takes !Other[i32]::A = i32 as a claim parameter, but
// a hypothesis in scope reads no premise: the respelling is the coercion's to
// state, and the witness is refused.

!U = !trait.poly<0>

trait.trait private @X(%self: !trait.claim<@X[!U]>) {}

trait.trait private @Other(%self: !trait.claim<@Other[!U]>) {
  trait.assoc_type @A
}

trait.trait private @Has(%self: !trait.claim<@Has[!U]>) {
  trait.assoc_type @Out
}

trait.impl private @Has_tuple(%self: !trait.claim<@Has[tuple<!U>]>, %x: !trait.claim<@X[!U]>) {
  trait.assoc_type @Out = i64
}

func.func @f(
    %v: !trait.proj<@Has[tuple<i32>], "Out">,
    %x: !trait.claim<@X[!trait.proj<@Other[i32], "A">]>,
    %e: !trait.claim<!trait.proj<@Other[i32], "A"> = i32>
) -> i64 {
  // expected-error @below {{premise 0 of impl '@Has_tuple' is '!trait.claim<@X[i32]>', and the witness supplies '!trait.claim<@X[!trait.proj<@Other[i32], "A">]>'}}
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple[i32] given(%x)
    : (!trait.claim<@X[!trait.proj<@Other[i32], "A">]>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
