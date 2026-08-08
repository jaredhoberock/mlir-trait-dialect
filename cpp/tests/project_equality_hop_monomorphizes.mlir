// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The hop result flows into a trait.coerce at a concrete instance and the whole
// module monomorphizes clean. The projected equality
// !trait.proj<@Has[i32], "Out"> = i64 is never proven, so impl selection leaves
// it; its endpoints ground-resolve to i64 = i64 through @Has_i32, discharging
// the leftover check, and the coerce folds. Nothing residual survives.

!S = !trait.poly<0>

trait.trait @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64] {
  trait.assoc_type @Out
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

// CHECK: func.func @m(%arg0: i64) -> i64
// CHECK-NEXT: return %arg0 : i64
func.func @m(%v: !trait.proj<@Has[i32], "Out">) -> i64 {
  %w = trait.witness @Has_i32 for @Has[i32]
  %e = trait.project %w : @Has[i32] by @Has_i32 to !trait.proj<@Has[i32], "Out"> = i64
  %c = trait.coerce %v : !trait.proj<@Has[i32], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@Has[i32], "Out"> = i64>)
  return %c : i64
}
