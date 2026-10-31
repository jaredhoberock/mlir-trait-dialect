// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Round-trip test for GAT (Generic Associated Type) declarations and projections.

// CHECK-LABEL: trait private @Test
// CHECK-NEXT: assoc {Wrapper = 1}
// CHECK-NEXT: members [
// CHECK-NEXT: @Test.test{{$}}
// CHECK-NEXT: ]
// CHECK: func.func private @Test.test(!trait.claim<@Test[!trait.poly<0>]>, !trait.poly<0>, !trait.poly<1>) -> !trait.proj<@Test[!trait.poly<0>], "Wrapper", [!trait.poly<1>]>

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Test(!trait.claim<@Test[!S]>) assoc {Wrapper = 1} members [@Test.test]
func.func private @Test.test(!trait.claim<@Test[!S]>, !S, !T) -> !trait.proj<@Test[!S], "Wrapper", [!T]>

// CHECK-LABEL: trait.impl private @Test_impl(%self: !trait.claim<@Test[i1]>)
// CHECK-NEXT: bindings {Wrapper = !trait.poly<0>}
// CHECK-NEXT: members [
// CHECK-NEXT: @Test.test -> @Test_impl.test{{$}}
// CHECK-NEXT: ]
// CHECK: func.func private @Test_impl.test

trait.impl private @Test_impl(%self_claim: !trait.claim<@Test[i1]>) bindings {Wrapper = !trait.poly<0>} members [@Test.test -> @Test_impl.test] {
}
func.func private @Test_impl.test(%self_claim: !trait.claim<@Test[i1]>, %self: i1, %value: !trait.poly<0>) -> !trait.poly<0> {
  return %value : !trait.poly<0>
}

// An associated type with no parameters of its own counts none
// CHECK-LABEL: trait private @Iterator
// CHECK-NEXT: assoc {Item = 0}
// CHECK-NEXT: members [
// CHECK-NEXT: @Iterator.next{{$}}
// CHECK-NEXT: ]
// CHECK: func.func private @Iterator.next(!trait.claim<@Iterator[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Iterator[!trait.poly<0>], "Item">

!U = !trait.poly<0>
trait.trait private @Iterator(!trait.claim<@Iterator[!U]>) assoc {Item = 0} members [@Iterator.next]
func.func private @Iterator.next(!trait.claim<@Iterator[!U]>, !U) -> !trait.proj<@Iterator[!U], "Item">
