// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

// A projection over a concrete base has a determined resolution. When no impl
// binds @T[i64]::A, the resolution patterns cannot resolve the projection, and
// instantiate-monomorphs rejects it, attributed to the op that carries it -- a
// loud diagnostic ahead of the legalization failure the leftover projection
// triggers downstream, rather than surfacing only as that opaque failure.

trait.trait @T[!trait.poly<0>] {
  trait.assoc_type @A
}

func.func @main() -> !trait.proj<@T[i64], "A"> {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@T[i64], "A">
  return %r : !trait.proj<@T[i64], "A">
}
