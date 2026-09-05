// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s

// An equality's endpoints are reachable to every walk and a leaf to every
// rewrite. The two rewrites the stage runs over the whole module are what that
// distinction is for: the proof stamper, which would stamp a proof into an
// endpoint and make it exactly the state the equality constructor refuses; and
// the ground-projection resolution, which would move a stored witness's own
// equality out from under the claim it must match.

trait.trait private @T[!trait.poly<0>] {
}

trait.impl private @T_i32 for @T[i32] {
}

// The stage records a proof for @T[i32] here, so the stamper has one to write.
func.func @demand() {
  %e = trait.allege @T[i32]
  return
}

// The claim standing in the endpoint keeps its unproven spelling.
// CHECK-LABEL: func.func private @holds
// CHECK-SAME: !trait.claim<tuple<!trait.claim<@T[i32]>> = tuple<!trait.claim<@T[i32]>>>
func.func private @holds(%c: !trait.claim<tuple<!trait.claim<@T[i32]>> = tuple<!trait.claim<@T[i32]>>>) {
  return
}

// -----

trait.trait private @Has[!trait.poly<0>] {
  trait.assoc_type @Out
}

trait.impl private @Has_i32 for @Has[i32] {
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
