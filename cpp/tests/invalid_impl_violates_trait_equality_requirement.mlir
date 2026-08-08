// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// @FoldFn requires Self::Output = Self, but the impl for i32 binds Output to
// i64. The projection Self::Output resolves through the impl's binding to i64,
// which is not i32, so the impl fails its birth check.

!S = !trait.poly<0>

trait.trait @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output
}

// CHECK: does not satisfy trait-header equality requirement
trait.impl @FoldFn_bad for @FoldFn[i32] {
  trait.assoc_type @Output = i64
}
