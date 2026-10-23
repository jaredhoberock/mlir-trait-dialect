// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// A projection over a concrete base has a determined resolution. When no impl
// binds @T[i64]::A, the resolution patterns cannot resolve the projection, and
// instantiate-monomorphs rejects it at each op that carries it -- the function
// whose declared result type names it, where selection's refusal is named, and
// the op producing it -- a loud diagnostic ahead of the legalization failure
// the leftover projection triggers downstream, rather than surfacing only as
// that opaque failure.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {
  trait.assoc_type @A
}

// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@T[i64], "A">'}}
// expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
func.func @main() -> !trait.proj<@T[i64], "A"> {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@T[i64], "A">
  return %r : !trait.proj<@T[i64], "A">
}
