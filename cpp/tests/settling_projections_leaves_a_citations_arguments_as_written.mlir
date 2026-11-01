// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// The stage resolves a ground projection an op spells in place, but leaves a
// citation's arguments as written, as it leaves the claim they are spelled
// with (`addPredicateRules`): the witness cites `@J` at the projection its
// equality spells, and resolving the argument alone to `i64` would cite `@J`
// at another application than the one resolved, which the witness's verifier
// refuses.

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) assoc {A = 0}
trait.impl private @I(%self: !trait.claim<@T[i64]>) bindings {A = i64} {
}
trait.trait private @U(!trait.claim<@U[!trait.poly<0>]>) assoc {B = 0}
trait.impl private @J(%self: !trait.claim<@U[tuple<!trait.poly<0>>]>) bindings {B = i1} {
}

// CHECK-LABEL: func.func @wit
// CHECK: trait.witness proj_resolve !trait.proj<@U[tuple<!trait.proj<@T[i64], "A">>], "B"> resolves i1 by @J[!trait.proj<@T[i64], "A">]
func.func @wit() -> !trait.claim<!trait.proj<@U[tuple<!trait.proj<@T[i64], "A">>], "B"> = i1> {
  %e = trait.witness proj_resolve !trait.proj<@U[tuple<!trait.proj<@T[i64], "A">>], "B"> resolves i1 by @J[!trait.proj<@T[i64], "A">]
    : !trait.claim<!trait.proj<@U[tuple<!trait.proj<@T[i64], "A">>], "B"> = i1>
  return %e : !trait.claim<!trait.proj<@U[tuple<!trait.proj<@T[i64], "A">>], "B"> = i1>
}
