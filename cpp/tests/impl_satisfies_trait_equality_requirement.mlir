// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// @FoldFn requires Self::Output = Self. The impl for i32 binds Output to i32, so
// the projection Self::Output resolves through the impl's own binding to i32,
// the two endpoints meet, and the impl verifies clean.

!S = !trait.poly<0>

trait.trait @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output
}

// CHECK: trait.impl @FoldFn_i32 for @FoldFn[i32]
trait.impl @FoldFn_i32 for @FoldFn[i32] {
  trait.assoc_type @Output = i32
}
