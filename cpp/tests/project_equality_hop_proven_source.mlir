// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The equality hop, from a proven and from an unproven source. An equality
// claim is never proven, so requirement 0 of the proven claim of @Has[i32] is
// !trait.proj<@Has[i32], "Out"> = i64 with no proof on it, exactly as it is for
// the unproven claim (@f): the requirement is instantiated at the source
// application either way, and only an application requirement reads a provider
// out of the source's evidence.

!S = !trait.poly<0>

trait.trait private @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64] {
  trait.assoc_type @Out
}

trait.impl private @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @g
// CHECK: trait.project %arg0[0] : <@Has[i32] by @Has_i32> -> <!trait.proj<@Has[i32], "Out"> = i64>
func.func @g(%p: !trait.claim<@Has[i32] by @Has_i32>) -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64> {
  %e = trait.project %p[0] : !trait.claim<@Has[i32] by @Has_i32> -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
}

// CHECK-LABEL: func.func @f
// CHECK: trait.project %arg0[0] : <@Has[i32]> -> <!trait.proj<@Has[i32], "Out"> = i64>
func.func @f(%p: !trait.claim<@Has[i32]>) -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64> {
  %e = trait.project %p[0] : !trait.claim<@Has[i32]> -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
}
