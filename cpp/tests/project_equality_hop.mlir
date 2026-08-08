// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The projection hop to a trait's equality requirement. @Has requires
// Self::Out = i64, so a claim of @Has[i32] projects to the equality
// !trait.proj<@Has[i32], "Out"> = i64 -- the requirement specialized at the
// source application. The source is unproven, the equality result is never
// proven, and the candidate set membership accepts it.

!S = !trait.poly<0>

trait.trait @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64] {
  trait.assoc_type @Out
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @f
// CHECK: trait.project %arg0: @Has[i32] to !trait.proj<@Has[i32], "Out"> = i64
func.func @f(%p: !trait.claim<@Has[i32]>) -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64> {
  %e = trait.project %p : @Has[i32] to !trait.proj<@Has[i32], "Out"> = i64
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
}
