// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Round-trip test for GAT (Generic Associated Type) declarations and projections.

// CHECK-LABEL: trait private @Test
// CHECK: trait.assoc_type @Wrapper<[!trait.poly<1>]>
// CHECK: trait.method @test(!trait.poly<0>, !trait.poly<1>) -> !trait.proj<@Test[!trait.poly<0>], "Wrapper", [!trait.poly<1>]>

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Test(%self: !trait.claim<@Test[!S]>) {
  trait.assoc_type @Wrapper<[!T]>
  trait.method @test(!S, !T) -> !trait.proj<@Test[!S], "Wrapper", [!T]>
}

// CHECK-LABEL: trait.impl private @Test_impl(%self: !trait.claim<@Test[i1]>
// CHECK: trait.assoc_type @Wrapper<[!trait.poly<1>]> = !trait.poly<1>
// CHECK: trait.method @test

trait.impl private @Test_impl(%self_claim: !trait.claim<@Test[i1]>) {
  trait.assoc_type @Wrapper<[!T]> = !T
  trait.method @test(%self: i1, %value: !T) -> !T {
    trait.return %value : !T
  }
}

// Non-GAT associated type still works without type_params
// CHECK-LABEL: trait private @Iterator
// CHECK: trait.assoc_type @Item
// CHECK: trait.method @next(!trait.poly<0>) -> !trait.proj<@Iterator[!trait.poly<0>], "Item">

!U = !trait.poly<0>
trait.trait private @Iterator(%self: !trait.claim<@Iterator[!U]>) {
  trait.assoc_type @Item
  trait.method @next(!U) -> !trait.proj<@Iterator[!U], "Item">
}
