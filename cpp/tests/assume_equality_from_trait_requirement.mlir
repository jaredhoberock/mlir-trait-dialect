// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait whose requirement is an equality states it among its results, and a
// default-method body reads it off the trait's self claim by position,
// spelling the equality claim the requirement states.

!S = !trait.poly<0>

trait.trait private @FoldFn(%self: !trait.claim<@FoldFn[!S]>) -> !trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S> {
  trait.assoc_type @Output
  trait.method @fold(%x: !S) -> !S {
    %e = trait.project %self[0] : !trait.claim<@FoldFn[!S]> -> !trait.claim<!trait.proj<@FoldFn[!S], "Output"> = !S>
    trait.return %x : !S
  }
}

// CHECK: trait.trait private @FoldFn(%self: !trait.claim<@FoldFn[!trait.poly<0>]>) -> !trait.claim<!trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>>
// CHECK: trait.project %self[0] : <@FoldFn[!trait.poly<0>]> -> <!trait.proj<@FoldFn[!trait.poly<0>], "Output"> = !trait.poly<0>>
