// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// The net for impl_method_source_spelled_projection.mlir: an impl method that
// keeps a SOURCE-spelled projection is checked through the impl's own bindings,
// not tolerated. Here the trait requires the second parameter to be Self (i32
// for this impl), but the method spells it as @Container[i32]::Elem, which the
// impl binds to i64. Resolving the source spelling through the impl's bindings
// exposes the disagreement as a located hard failure -- the binding-resolution
// path carries the weight, with no lenient projection-vs-concrete crossing to
// absorb it.

!S = !trait.poly<0>

trait.trait @Container[!S] {
  trait.assoc_type @Elem
  func.func private @id(!S, !S) -> !S
}

// CHECK: op type mismatch: expected 'i32' but found 'i64'
// CHECK: method 'id' has incompatible signature
trait.impl for @Container[i32] {
  trait.assoc_type @Elem = i64
  func.func @id(%self: i32, %e: !trait.proj<@Container[i32], "Elem">) -> i32 {
    return %self : i32
  }
}
