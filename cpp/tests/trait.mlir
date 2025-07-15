// RUN: mlir-opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait @Add
// CHECK: func.func private @add(!trait.poly<0>, !trait.poly<0>) -> !trait.poly<0>

!AddSelf = !trait.poly<0>
trait.trait @Add[!AddSelf] {
  func.func private @add(!AddSelf, !AddSelf) -> !AddSelf
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait @PartialEq
// CHECK: func.func private @eq(!trait.poly<1>, !trait.poly<2>) -> i1
// CHECK: func.func private @neq(!trait.poly<1>, !trait.poly<2>) -> i1

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
trait.trait @PartialEq[!PartialEqSelf, !PartialEqOther] {
  func.func private @eq(!PartialEqSelf, !PartialEqOther) -> i1
  func.func private @neq(!PartialEqSelf, !PartialEqOther) -> i1
}
