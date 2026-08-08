// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// The enclosing trait declares no equality requirement and the method takes no
// equality parameter, so the equality assumed here matches no axiom of the
// enclosing scope. Strict verification refuses it; an equality assume is never
// accepted without an anchor.

!S = !trait.poly<0>

trait.trait @FoldFn[!S] {
  trait.assoc_type @Output
  func.func @fold(%x: !S) -> !S {
    // CHECK: assumed equality {{.*}} is not assumable in this context
    %e = trait.assume !trait.proj<@FoldFn[!S], "Output"> = !S
    return %x : !S
  }
}
