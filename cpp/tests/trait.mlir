// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait private @Add
// CHECK: trait.method @add(!trait.poly<0>, !trait.poly<0>) -> !trait.poly<0>

!AddSelf = !trait.poly<0>
trait.trait private @Add(%self: !trait.claim<@Add[!AddSelf]>) {
  trait.method @add(!AddSelf, !AddSelf) -> !AddSelf
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait private @PartialEq
// CHECK: trait.method @eq(!trait.poly<0>, !trait.poly<1>) -> i1
// CHECK: trait.method @neq(%{{.*}}: !trait.poly<0>, %{{.*}}: !trait.poly<1>) -> i1

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
trait.trait private @PartialEq(%self_claim: !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>) {
  trait.method @eq(!trait.poly<0>, !trait.poly<1>) -> i1
  
  trait.method @neq(%self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
    %eq = trait.method.call %self_claim @PartialEq[!trait.poly<0>,!trait.poly<1>]::@eq(%self, %other)
      : (!trait.poly<0>, !trait.poly<1>) -> i1

    %true = arith.constant true
    %res = arith.xori %eq, %true : i1
    trait.return %res : i1
  }
}

// ---- Test 2: PartialOrd

// CHECK-LABEL: trait private @PartialOrd
// CHECK: trait.method @partial_cmp(!trait.poly<0>, !trait.poly<1>) -> !llvm.struct<"ordering", ()>
// CHECK: trait.method @lt(%{{.*}}: !trait.poly<0>, %{{.*}}: !trait.poly<1>) -> i1

!ordering = !llvm.struct<"ordering", ()>
!PartialOrdSelf = !trait.poly<3>
!PartialOrdOther = !trait.poly<4>
trait.trait private @PartialOrd(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]> {
  trait.method @partial_cmp(!trait.poly<0>, !trait.poly<1>) -> !ordering

  trait.method @lt(%self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
    %cmp = trait.method.call %self_claim @PartialOrd[!trait.poly<0>,!trait.poly<1>]::@partial_cmp(%self, %other)
      : (!trait.poly<0>, !trait.poly<1>) -> !ordering

    %res = arith.constant false
    trait.return %res : i1
  }
}
