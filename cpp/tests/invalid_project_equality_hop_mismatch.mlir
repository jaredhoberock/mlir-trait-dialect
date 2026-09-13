// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// @Has requires Self::Out = i64, so requirement 0 of @Has[i32] is
// !trait.proj<@Has[i32], "Out"> = i64. The result type the op spells is an
// annotation on that selection, and this one disagrees at the right endpoint.

!S = !trait.poly<0>

trait.trait private @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64] {
  trait.assoc_type @Out
}

trait.impl private @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

func.func @f(%p: !trait.claim<@Has[i32]>) -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i32> {
  // CHECK: type mismatch: expected '!trait.claim<!trait.proj<@Has[i32], "Out"> = i64>' but found '!trait.claim<!trait.proj<@Has[i32], "Out"> = i32>'
  %e = trait.project %p[0] : !trait.claim<@Has[i32]> -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i32>
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i32>
}
