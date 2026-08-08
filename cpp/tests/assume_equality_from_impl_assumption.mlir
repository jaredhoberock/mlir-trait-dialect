// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An impl method body mints an equality-arm trait.assume whose equality matches,
// by identity, one of the enclosing impl's own equality assumptions. The impl's
// assumption list anchors the equality assume exactly as it anchors an
// application assume, so the assume is legal and the module verifies.

!S = !trait.poly<0>
!Acc = !trait.poly<1>

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
  func.func private @fold(!S) -> !S
}

// CHECK-LABEL: trait.impl @FoldFn_gen for @FoldFn[!trait.poly<0>]
// CHECK: trait.assume !trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<1>
trait.impl @FoldFn_gen for @FoldFn[!S] where [!trait.proj<@FoldFn[!S], "Output"> = !Acc] {
  trait.assoc_type @Output = !Acc
  func.func @fold(%x: !S) -> !S {
    %e = trait.assume !trait.proj<@FoldFn[!S], "Output"> = !Acc
    return %x : !S
  }
}
