// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A witness names @Box_i64, which proves @Box[i64], for a claim spelling
// @Box[@Gen[i64]::A]. A witness carries the application its declaration
// proves, so it is refused where it stands; a respelling is a coercion citing
// the projection's binding, which the sibling row writes once @Gen binds it.

trait.trait private @Gen(!trait.claim<@Gen[!trait.poly<0>]>) assoc {A = 0}

trait.trait private @Box(!trait.claim<@Box[!trait.poly<0>]>) members [@Box.v]
func.func private @Box.v(!trait.claim<@Box[!trait.poly<0>]>) -> i64

trait.impl private @Box_i64(%self: !trait.claim<@Box[i64]>) members [@Box.v -> @Box_i64.v] {
}
func.func private @Box_i64.v(%self: !trait.claim<@Box[i64]>) -> i64 {
  %c = arith.constant 64 : i64
  return %c : i64
}

func.func @main() -> i64 {
  // expected-error @below {{proof @Box_i64 proves '!trait.claim<@Box[i64]>', which does not discharge the obligation '!trait.claim<@Box[!trait.proj<@Gen[i64], "A">]>'}}
  %w = trait.witness @Box_i64 for @Box[!trait.proj<@Gen[i64], "A">]
  %r = trait.func.call @Box.v(%w) : (!trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i64>) -> i64
  return %r : i64
}
