// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s --emit-bytecode | mlir-opt | FileCheck %s

// An impl's `where` clause carries a mix of arms -- an application assumption
// and an equality assumption -- in one ordered predicate list, in declaration
// order. There is no second array. The clause survives both the textual and the
// bytecode round-trip with its entries and their order intact.

!S = !trait.poly<0>

trait.trait @Eq[!S] {}

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
}

// CHECK: trait.impl @FoldFn_cond for @FoldFn[!trait.poly<0>]where [@Eq[!trait.poly<0>], !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>]
trait.impl @FoldFn_cond for @FoldFn[!S] where [@Eq[!S], !trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output = !S
}
