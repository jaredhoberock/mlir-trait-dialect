// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// An impl asserts its own equality Self::Output = i32 but binds Output to i64.
// The projection resolves through the impl's own binding to i64, which is not
// i32, so the impl fails its own-assumption equality birth check.

!S = !trait.poly<0>

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
}

// CHECK: does not satisfy its own equality predicate
trait.impl @FoldFn_bad for @FoldFn[i32] where [!trait.proj<@FoldFn[i32], "Output"> = i32] {
  trait.assoc_type @Output = i64
}
