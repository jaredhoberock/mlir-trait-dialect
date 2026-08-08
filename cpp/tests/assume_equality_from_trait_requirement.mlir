// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait default-method body mints an equality-arm trait.assume whose equality
// matches, by identity, one of the enclosing trait's equality requirements. The
// trait's requirement list anchors the equality assume, so the assume is legal
// and the module verifies.

!S = !trait.poly<0>

trait.trait @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !S] {
  trait.assoc_type @Output
  func.func @fold(%x: !S) -> !S {
    %e = trait.assume !trait.proj<@FoldFn[!S], "Output"> = !S
    return %x : !S
  }
}

// CHECK: trait.trait @FoldFn[!trait.poly<0>] where [!trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>]
// CHECK: trait.assume !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>
