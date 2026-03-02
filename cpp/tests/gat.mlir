// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Round-trip test for GAT (Generic Associated Type) declarations and projections.

// CHECK-LABEL: trait @Test
// CHECK: trait.assoc_type @Wrapper<[!trait.poly<1>]>
// CHECK: func.func private @test(!trait.poly<0>, !trait.poly<1>) -> !trait.proj<@Test[!trait.poly<0>], "Wrapper", [!trait.poly<1>]>

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @Test[!S] {
  trait.assoc_type @Wrapper<[!T]>
  func.func private @test(!S, !T) -> !trait.proj<@Test[!S], "Wrapper", [!T]>
}

// CHECK-LABEL: trait.impl for @Test[i1]
// CHECK: trait.assoc_type @Wrapper<[!trait.poly<1>]> = !trait.poly<1>
// CHECK: func.func @test

trait.impl for @Test[i1] {
  trait.assoc_type @Wrapper<[!T]> = !T
  func.func @test(%self: i1, %value: !T) -> !T {
    return %value : !T
  }
}

// Non-GAT associated type still works without type_params
// CHECK-LABEL: trait @Iterator
// CHECK: trait.assoc_type @Item
// CHECK: func.func private @next(!trait.poly<0>) -> !trait.proj<@Iterator[!trait.poly<0>], "Item">

!U = !trait.poly<0>
trait.trait @Iterator[!U] {
  trait.assoc_type @Item
  func.func private @next(!U) -> !trait.proj<@Iterator[!U], "Item">
}
