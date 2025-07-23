// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: test everything

// CHECK-LABEL: trait @PartialEq [!trait.poly<0>, !trait.poly<1>]
// CHECK-LABEL: func.func private @eq
// CHECK-LABEL: func.func private @ne
!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait @PartialEq[!S,!O] {
  func.func private @eq(!S, !O) -> i1
  func.func private @ne(!S, !O) -> i1
}

// CHECK-LABEL impl @PartialEq [i32,i32]
trait.impl @PartialEq[i32,i32] {
  // CHECK-LABEL: func @eq
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    return %equal : i1
  }

  // CHECK-LABEL: func @ne
  func.func @ne(%self: i32, %other: i32) -> i1 {
    %w = trait.witness : !trait.proof<@PartialEq[i32,i32]>
    %equal = trait.method.call @PartialEq::@eq<%w>(%self, %other)
      : (!S, !O) -> i1
      as !trait.proof<@PartialEq[i32,i32]> (i32,i32) -> i1
    %true = arith.constant 1 : i1
    %not_equal = arith.xori %equal, %true : i1
    return %not_equal : i1
  }
}


// CHECK-LABEL: func @foo
!T = !trait.poly<0>
!W = !trait.proof<@PartialEq[!T,!T]>
func.func @foo(%w: !W, %x: !T, %y: !T) -> i1 {
  // CHECK: %[[RES:.*]] = trait.method.call @PartialEq
  %res = trait.method.call @PartialEq::@eq<%w>(%x, %y)
    : (!S, !O) -> i1
    as !W (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func @bar
func.func @bar(%x: i32, %y: i32) -> i1 {
  %w = trait.witness : !trait.proof<@PartialEq[i32,i32]>

  // CHECK: %[[RES:.*]] = trait.func.call @foo
  %res = trait.func.call @foo(%w, %x, %y)
    : (!W,!T,!T) -> i1
    as (!trait.proof<@PartialEq[i32,i32]>, i32, i32) -> i1

  return %res : i1
}
