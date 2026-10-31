// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait.trait private @Add(!trait.claim<@Add[!trait.poly<0>]>) members [@Add.add]
// CHECK: func.func private @Add.add(!trait.claim<@Add[!trait.poly<0>]>, !trait.poly<0>, !trait.poly<0>) -> !trait.poly<0>

!AddSelf = !trait.poly<0>
trait.trait private @Add(!trait.claim<@Add[!AddSelf]>) members [@Add.add]
func.func private @Add.add(!trait.claim<@Add[!AddSelf]>, !AddSelf, !AddSelf) -> !AddSelf

// ---- Test 1: PartialEq

// CHECK-LABEL: trait.trait private @PartialEq(!trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>) members [@PartialEq.eq, default @PartialEq.neq]
// CHECK: func.func private @PartialEq.eq(!trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> i1
// CHECK: func.func private @PartialEq.neq(%{{.*}}: !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>, %{{.*}}: !trait.poly<0>, %{{.*}}: !trait.poly<1>) -> i1

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
trait.trait private @PartialEq(!trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>) members [@PartialEq.eq, default @PartialEq.neq]
func.func private @PartialEq.eq(!trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> i1
func.func private @PartialEq.neq(%self_claim: !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]>, %self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
  %eq = trait.func.call @PartialEq.eq(%self_claim, %self, %other) : (!trait.claim<@PartialEq[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> i1

  %true = arith.constant true
  %res = arith.xori %eq, %true : i1
  return %res : i1
}

// ---- Test 2: PartialOrd

// CHECK-LABEL: trait.trait private @PartialOrd(!trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]> members [@PartialOrd.partial_cmp, default @PartialOrd.lt]
// CHECK: func.func private @PartialOrd.partial_cmp(!trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> !llvm.struct<"ordering", ()>
// CHECK: func.func private @PartialOrd.lt(%{{.*}}: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, %{{.*}}: !trait.poly<0>, %{{.*}}: !trait.poly<1>) -> i1

!ordering = !llvm.struct<"ordering", ()>
!PartialOrdSelf = !trait.poly<3>
!PartialOrdOther = !trait.poly<4>
trait.trait private @PartialOrd(!trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>) -> !trait.claim<@PartialEq[!trait.poly<0>, !trait.poly<1>]> members [@PartialOrd.partial_cmp, default @PartialOrd.lt]
func.func private @PartialOrd.partial_cmp(!trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> !ordering
func.func private @PartialOrd.lt(%self_claim: !trait.claim<@PartialOrd[!trait.poly<0>, !trait.poly<1>]>, %self: !trait.poly<0>, %other: !trait.poly<1>) -> i1 {
  %cmp = trait.func.call @PartialOrd.partial_cmp(%self_claim, %self, %other) : (!trait.claim<@PartialOrd[!trait.poly<0>,!trait.poly<1>]>, !trait.poly<0>, !trait.poly<1>) -> !ordering

  %res = arith.constant false
  return %res : i1
}
