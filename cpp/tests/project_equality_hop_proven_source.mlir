// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The equality hop from a PROVEN source. Proofness parity would force a proven
// source to project to a proven result, but an equality claim is never proven,
// so the equality arm is exempt: the proven claim of @Has[i32] projects to
// !trait.proj<@Has[i32], "Out"> = i64 with no receipt on the result.

!S = !trait.poly<0>

trait.trait @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64] {
  trait.assoc_type @Out
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @g
// CHECK: trait.project %arg0: @Has[i32] by @Has_i32 to !trait.proj<@Has[i32], "Out"> = i64
func.func @g(%p: !trait.claim<@Has[i32] by @Has_i32>) -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64> {
  %e = trait.project %p : @Has[i32] by @Has_i32 to !trait.proj<@Has[i32], "Out"> = i64
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
}
