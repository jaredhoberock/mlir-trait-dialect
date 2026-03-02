// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: trait @Iterator
// CHECK: trait.assoc_type @Item
// CHECK: func.func private @next(!trait.poly<0>) -> !trait.proj<@Iterator[!trait.poly<0>], "Item">

!S = !trait.poly<0>
trait.trait @Iterator[!S] {
  trait.assoc_type @Item
  func.func private @next(!S) -> !trait.proj<@Iterator[!S], "Item">
}

// CHECK-LABEL: trait.impl for @Iterator[i32]
// CHECK: trait.assoc_type @Item = i64
// CHECK: func.func @next

trait.impl for @Iterator[i32] {
  trait.assoc_type @Item = i64
  func.func @next(%self: i32) -> i64 {
    %c = arith.constant 42 : i64
    return %c : i64
  }
}
