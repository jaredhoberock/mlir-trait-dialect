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
// CHECK: func.func @neq(%{{.*}}: !trait.poly<1>, %{{.*}}: !trait.poly<2>) -> i1

!PartialEqSelf = !trait.poly<1>
!PartialEqOther = !trait.poly<2>
trait.trait @PartialEq[!PartialEqSelf, !PartialEqOther] {
  func.func private @eq(!PartialEqSelf, !PartialEqOther) -> i1
  
  func.func @neq(%self: !PartialEqSelf, %other: !PartialEqOther) -> i1 {
    %partial_eq = trait.assume @PartialEq[!PartialEqSelf, !PartialEqOther]

    %eq = trait.method.call %partial_eq @PartialEq[!PartialEqSelf,!PartialEqOther]::@eq(%self, %other)
      :  (!PartialEqSelf, !PartialEqOther) -> i1
      as (!PartialEqSelf, !PartialEqOther) -> i1

    %true = arith.constant true
    %res = arith.xori %eq, %true : i1
    return %res : i1
  }
}

// ---- Test 2: PartialOrd

// CHECK-LABEL: trait @PartialOrd
// CHECK: func.func private @partial_cmp(!trait.poly<3>, !trait.poly<4>) -> !llvm.struct<"ordering", ()>
// CHECK: func.func @lt(%{{.*}}: !trait.poly<3>, %{{.*}}: !trait.poly<4>) -> i1

!ordering = !llvm.struct<"ordering", ()>
!PartialOrdSelf = !trait.poly<3>
!PartialOrdOther = !trait.poly<4>
trait.trait @PartialOrd[!PartialOrdSelf, !PartialOrdOther] where [
  @PartialEq[!PartialOrdSelf, !PartialOrdOther]
]
{
  func.func private @partial_cmp(!PartialOrdSelf, !PartialOrdOther) -> !ordering

  func.func @lt(%self: !PartialOrdSelf, %other: !PartialOrdOther) -> i1 {
    %partial_ord = trait.assume @PartialOrd[!PartialOrdSelf,!PartialOrdOther]

    %cmp = trait.method.call %partial_ord @PartialOrd[!PartialOrdSelf,!PartialOrdOther]::@partial_cmp(%self, %other)
      :  (!PartialOrdSelf, !PartialOrdOther) -> !ordering
      as (!PartialOrdSelf, !PartialOrdOther) -> !ordering

    %res = arith.constant false
    return %res : i1
  }
}
