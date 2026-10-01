// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait default-method body cites one of the enclosing trait's equality
// requirements by position, spelling the equality claim the requirement
// states, so the assume is legal and the module verifies.

!S = !trait.poly<0>

trait.trait private @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output
  trait.method @fold(%x: !S) -> !S {
    %e = trait.assume 0 : !trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S>
    trait.return %x : !S
  }
}

// CHECK: trait.trait private @FoldFn[!trait.poly<0>] where [!trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>]
// CHECK: trait.assume 0 : !trait.claim<!trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>>
