// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// End-to-end monomorphization test for GAT projections.
//
// trait Test {
//   type Wrapper<T>;
//   fn test<T>(self: Self, other: T) -> Self::Wrapper<T>;
// }
// impl Test for Bool {
//   type Wrapper<T> = T;
//   fn test<T>(self: Self, value: T) -> T { value }
// }

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Test(%self: !trait.claim<@Test[!S]>) {
  trait.assoc_type @Wrapper<[!T]>
  trait.method @test(!S, !T) -> !trait.proj<@Test[!S], "Wrapper", [!T]>
}

trait.impl private @Test_impl(%self_claim: !trait.claim<@Test[i1]>) {
  trait.assoc_type @Wrapper<[!T]> = !T
  trait.method @test(%self: i1, %value: !T) -> !T {
    trait.return %value : !T
  }
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: !trait.proj
// CHECK: call @
// CHECK-SAME: (i1, i32) -> i32
// CHECK: return %{{.*}} : i32
func.func @caller() -> !trait.proj<@Test[i1], "Wrapper", [i32]> {
  %a = trait.allege @Test[i1]
  %self = arith.constant 1 : i1
  %x = arith.constant 42 : i32
  %r = trait.method.call %a @Test[i1]::@test(%self, %x)
    : (i1, i32) -> !trait.proj<@Test[i1], "Wrapper", [i32]>
  return %r : !trait.proj<@Test[i1], "Wrapper", [i32]>
}
