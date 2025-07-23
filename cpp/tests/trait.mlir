// RUN: mlir-opt %s | FileCheck %s

// ---- Test 0: Add

// CHECK-LABEL: trait @Add
// CHECK: trait.method @add(!trait.poly<0>, !trait.poly<0>) -> !trait.poly<0>

!AddSelf = !trait.poly<0>
trait.trait @Add[!AddSelf] {
  trait.method @add(!AddSelf, !AddSelf) -> !AddSelf
}

// ---- Test 1: PartialEq

// CHECK-LABEL: trait @PartialEq
// CHECK: trait.method @eq(!trait.poly<1>, !trait.poly<2>) -> i1
// CHECK: trait.method @neq(%{{.*}}: !trait.witness<@PartialEq[!trait.poly<1>, !trait.poly<2>]>, %{{.*}}: !trait.poly<1>, %{{.*}}: !trait.poly<2>) -> i1 {
// CHECK: trait.method.call @PartialEq::@eq
// CHECK: arith.constant true
// CHECK: arith.xori
// CHECK: return
// CHECK: }

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
!PartialEqWitness = !trait.witness<@PartialEq[!PartialEqSelf, !PartialEqOther]>
trait.trait @PartialEq[!PartialEqSelf, !PartialEqOther] {
  trait.method @eq(!PartialEqSelf, !PartialEqOther) -> i1
  trait.method @neq(%w: !PartialEqWitness, %self: !PartialEqSelf, %other: !PartialEqOther) -> i1 {
    %equal = trait.method.call @PartialEq::@eq<%w>(%self, %other)
      : (!PartialEqSelf, !PartialEqOther) -> i1
      as !PartialEqWitness (!PartialEqSelf, !PartialEqOther) -> i1
    %true = arith.constant 1 : i1
    %result = arith.xori %equal, %true : i1
    return %result : i1
  }
}
