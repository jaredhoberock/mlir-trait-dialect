// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @use takes a claim over a projection of its parameter. The call supplies a
// proof cited by hand over @B[@A[i64]::Item], and no impl of @A serves i64, so
// the substitution the call's instance is cut with reads a projection that
// names no type, and the call cannot lower. It survives instantiation and is
// named there, with selection's refusal of the projection beside it.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.impl private @B_proj(%self: !trait.claim<@B[!trait.proj<@A[i64], "Item">]>) {}
func.func private @use(%w: !trait.claim<@B[!trait.proj<@A[!T], "Item">]>, %x: !T) -> !T {
  return %x : !T
}
func.func @main() -> i64 {
  %k = arith.constant 7 : i64
  %w = trait.witness @B_proj for @B[!trait.proj<@A[i64], "Item">]
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i64], "Item">'}}
  // expected-error @below {{'trait.func.call' op rewritable generic call survived instantiate-monomorphs}}
  %r = trait.func.call @use(%w, %k) : (!trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>, i64) -> i64
  return %r : i64
}
