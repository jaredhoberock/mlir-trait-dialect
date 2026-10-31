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

trait.trait private @Test(!trait.claim<@Test[!S]>) assoc {Wrapper = 1} members [@Test.test]
func.func private @Test.test(!trait.claim<@Test[!S]>, !S, !T) -> !trait.proj<@Test[!S], "Wrapper", [!T]>

trait.impl private @Test_impl(%self_claim: !trait.claim<@Test[i1]>) bindings {Wrapper = !trait.poly<0>} members [@Test.test -> @Test_impl.test] {
}
func.func private @Test_impl.test(%self_claim: !trait.claim<@Test[i1]>, %self: i1, %value: !trait.poly<0>) -> !trait.poly<0> {
  return %value : !trait.poly<0>
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
  %r = trait.func.call @Test.test(%a, %self, %x) : (!trait.claim<@Test[i1]>, i1, i32) -> !trait.proj<@Test[i1], "Wrapper", [i32]>
  return %r : !trait.proj<@Test[i1], "Wrapper", [i32]>
}
