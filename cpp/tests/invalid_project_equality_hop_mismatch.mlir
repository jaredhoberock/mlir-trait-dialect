// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// @Has requires Self::Out = i64, so !trait.proj<@Has[i32], "Out"> = i32 is not a
// candidate projection of @Has[i32]: the candidate-set comparison is exact, and
// the requested equality's right endpoint differs from the requirement's.

!S = !trait.poly<0>

trait.trait @Has[!S] where [!trait.proj<@Has[!S], "Out"> = i64] {
  trait.assoc_type @Out
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

func.func @f(%p: !trait.claim<@Has[i32]>) -> !trait.claim<!trait.proj<@Has[i32], "Out"> = i32> {
  // CHECK: is not a candidate projection
  %e = trait.project %p : @Has[i32] to !trait.proj<@Has[i32], "Out"> = i32
  return %e : !trait.claim<!trait.proj<@Has[i32], "Out"> = i32>
}
