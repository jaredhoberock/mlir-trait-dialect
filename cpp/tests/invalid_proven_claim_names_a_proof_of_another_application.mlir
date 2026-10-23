// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A proven claim names the proof of exactly the application it spells,
// wherever a value of it stands, so a call dispatching on the citation runs an
// impl of the claim it holds. Each claim below names @T_i64, which proves
// @T[i64], for @T[i32]: as a function's result, as the result of an op outside
// the dialect, and as a coercion's result carrying its input's proof to another
// spelling of its application.

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) {}
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {}
// expected-error @below {{proof @T_i64 proves '!trait.claim<@T[i64]>', which does not discharge the obligation '!trait.claim<@T[i32]>'}}
func.func private @f() -> !trait.claim<@T[i32] by @T_i64>

// -----

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) {}
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {}
func.func @f() {
  // expected-error @below {{proof @T_i64 proves '!trait.claim<@T[i64]>', which does not discharge the obligation '!trait.claim<@T[i32]>'}}
  %c = ub.poison : !trait.claim<@T[i32] by @T_i64>
  return
}

// -----

!T = !trait.poly<0>
trait.trait private @T(%self: !trait.claim<@T[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {}
trait.impl private @A_i8(%self: !trait.claim<@A[i8]>) { trait.assoc_type @Item = i64 }
func.func @f() {
  %w = trait.witness @T_i64 for @T[i64]
  %e = trait.witness proj_resolve !trait.proj<@A[i8], "Item"> resolves i64 by @A_i8
    : !trait.claim<!trait.proj<@A[i8], "Item"> = i64>
  // expected-error @below {{proof @T_i64 proves '!trait.claim<@T[i64]>', which does not discharge the obligation '!trait.claim<@T[!trait.proj<@A[i8], "Item">]>'}}
  %c = trait.coerce %w : !trait.claim<@T[i64] by @T_i64>
    to !trait.claim<@T[!trait.proj<@A[i8], "Item">] by @T_i64>
    via (%e) : (!trait.claim<!trait.proj<@A[i8], "Item"> = i64>)
  return
}
