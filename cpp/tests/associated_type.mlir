// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: trait private @Iterator
// CHECK: trait.assoc_type @Item
// CHECK: trait.method @next(!trait.poly<0>) -> !trait.proj<@Iterator[!trait.poly<0>], "Item">

!S = !trait.poly<0>
trait.trait private @Iterator(%self: !trait.claim<@Iterator[!S]>) {
  trait.assoc_type @Item
  trait.method @next(!S) -> !trait.proj<@Iterator[!S], "Item">
}

// CHECK-LABEL: trait.impl private @Iterator_impl(%self: !trait.claim<@Iterator[i32]>
// CHECK: trait.assoc_type @Item = i64
// CHECK: trait.method @next

trait.impl private @Iterator_impl(%self_claim: !trait.claim<@Iterator[i32]>) {
  trait.assoc_type @Item = i64
  trait.method @next(%self: i32) -> i64 {
    %c = arith.constant 42 : i64
    trait.return %c : i64
  }
}
