// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A premise spelled through a projection discharges the cited impl's where
// entry modulo an equality the citation's scope holds. @Has_tuple's entry is
// @X[i32] at the citation; the witness supplies @X[!Other[i32]::A], and @f
// takes !Other[i32]::A = i32 as a claim parameter. No impl binds
// @Other[i32]::A, so that hypothesis alone reads the supplied premise as
// @X[i32], and the entry is discharged.

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
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple given(%x)
    : (!trait.claim<@X[!trait.proj<@Other[i32], "A">]>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}

// -----

// Without the equality in scope nothing reads @X[!Other[i32]::A] as @X[i32],
// and the witness is refused.

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
    %x: !trait.claim<@X[!trait.proj<@Other[i32], "A">]>
) -> i64 {
  // expected-error @below {{premise 0 of impl '@Has_tuple' is '!trait.claim<@X[i32]>', and the witness supplies '!trait.claim<@X[!trait.proj<@Other[i32], "A">]>'}}
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple given(%x)
    : (!trait.claim<@X[!trait.proj<@Other[i32], "A">]>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
