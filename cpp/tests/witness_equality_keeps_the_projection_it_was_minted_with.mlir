// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// An equality's endpoints are reachable to every walk and a leaf to every
// rewrite. The ground-projection resolution the stage runs over the whole
// module is what that distinction is for: it would move a stored witness's own
// equality out from under the claim it must match.

trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) {
  trait.assoc_type @Out
}

trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @Out = i64
}

// The witness's stored equality keeps the ground projection it was minted with,
// so the attribute and the result claim still name the same equality and the
// op's verifier accepts it.
// CHECK: trait.witness proj_resolve !trait.proj<@Has[i32], "Out"> resolves i64 by @Has_i32
func.func @wit() -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@Has[i32], "Out"> resolves i64 by @Has_i32
    : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
}
