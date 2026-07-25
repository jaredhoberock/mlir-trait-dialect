// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' -verify-diagnostics

// A `by @proof` spelled in a function signature is verified where it is
// declared: the named proof must specialize to the claim it annotates. Here the
// signature claims @T[i64] but names @T_i32_p, which proves @T[i32]. The poly
// parameter keeps the function polymorphic so the monomorph-claim ban does not
// mask the real defect.

trait.trait @T[!trait.poly<0>] {}
trait.impl @T_i32 for @T[i32] {}
trait.proof @T_i32_p proves @T_i32 for @T[i32] given []

// expected-error @below {{declared claim in signature has an invalid proof}}
func.func private @f(%x: !trait.poly<9>, %c: !trait.claim<@T[i64] by @T_i32_p>) {
  return
}
