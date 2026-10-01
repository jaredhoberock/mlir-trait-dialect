// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait private @Add
// CHECK: trait.method @add(!trait.poly<0>, !trait.poly<0>) -> !trait.poly<0>

!AddSelf = !trait.poly<0>
trait.trait private @Add[!AddSelf] {
  trait.method @add(!AddSelf, !AddSelf) -> !AddSelf
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait private @PartialEq
// CHECK: trait.method @eq(!trait.poly<1>, !trait.poly<2>) -> i1
// CHECK: trait.method @neq(%{{.*}}: !trait.poly<1>, %{{.*}}: !trait.poly<2>) -> i1

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
trait.trait private @PartialEq[!PartialEqSelf, !PartialEqOther] {
  trait.method @eq(!PartialEqSelf, !PartialEqOther) -> i1
  
  trait.method @neq(%self: !PartialEqSelf, %other: !PartialEqOther) -> i1 {
    %partial_eq = trait.assume self : !trait.claim<@PartialEq[!PartialEqSelf, !PartialEqOther]>

    %eq = trait.method.call %partial_eq @PartialEq[!PartialEqSelf,!PartialEqOther]::@eq(%self, %other)
      : (!PartialEqSelf, !PartialEqOther) -> i1

    %true = arith.constant true
    %res = arith.xori %eq, %true : i1
    trait.return %res : i1
  }
}

// ---- Test 2: PartialOrd

// CHECK-LABEL: trait private @PartialOrd
// CHECK: trait.method @partial_cmp(!trait.poly<3>, !trait.poly<4>) -> !llvm.struct<"ordering", ()>
// CHECK: trait.method @lt(%{{.*}}: !trait.poly<3>, %{{.*}}: !trait.poly<4>) -> i1

!ordering = !llvm.struct<"ordering", ()>
!PartialOrdSelf = !trait.poly<3>
!PartialOrdOther = !trait.poly<4>
trait.trait private @PartialOrd[!PartialOrdSelf, !PartialOrdOther] where [
  @PartialEq[!PartialOrdSelf, !PartialOrdOther]
]
{
  trait.method @partial_cmp(!PartialOrdSelf, !PartialOrdOther) -> !ordering

  trait.method @lt(%self: !PartialOrdSelf, %other: !PartialOrdOther) -> i1 {
    %partial_ord = trait.assume self : !trait.claim<@PartialOrd[!PartialOrdSelf,!PartialOrdOther]>

    %cmp = trait.method.call %partial_ord @PartialOrd[!PartialOrdSelf,!PartialOrdOther]::@partial_cmp(%self, %other)
      : (!PartialOrdSelf, !PartialOrdOther) -> !ordering

    %res = arith.constant false
    trait.return %res : i1
  }
}
