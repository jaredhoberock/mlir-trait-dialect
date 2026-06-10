// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An impl method may keep its SOURCE spelling: a parameter or result written
// in terms of Self::Assoc lowers as a projection naming the impl's own trait
// application. The verifier resolves it through the impl's associated type
// bindings rather than requiring the pre-resolved spelling.
//
// NOTE: today this also verifies via the lenient projection-vs-concrete
// unification rule; this test is the regression net for when that rule is
// removed (the binding-resolution path must carry the weight alone).

// CHECK-LABEL: trait @Container
!S = !trait.poly<0>
trait.trait @Container[!S] {
  trait.assoc_type @Elem
  func.func private @id(!S, !trait.proj<@Container[!S], "Elem">) -> !trait.proj<@Container[!S], "Elem">
}

// CHECK-LABEL: trait.impl for @Container[i32]
// CHECK: func.func @id(%{{.*}}: i32, %{{.*}}: !trait.proj<@Container[i32], "Elem">) -> !trait.proj<@Container[i32], "Elem">
trait.impl for @Container[i32] {
  trait.assoc_type @Elem = i64
  // The method signature keeps the projection spelling instead of i64.
  func.func @id(%self: i32, %e: !trait.proj<@Container[i32], "Elem">) -> !trait.proj<@Container[i32], "Elem"> {
    return %e : !trait.proj<@Container[i32], "Elem">
  }
}
