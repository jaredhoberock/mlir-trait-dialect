// RUN: opt %s | FileCheck %s

// ---- Test 1: test everything

// CHECK-LABEL: trait @PartialEq
// CHECK-LABEL: func.func private @eq
// CHECK-LABEL: func.func private @neq
trait.trait @PartialEq {
  func.func private @eq(!trait.self, !trait.self) -> i1
  func.func private @neq(!trait.self, !trait.self) -> i1
}

// CHECK-LABEL impl @PartialEq for i32
trait.impl @PartialEq for i32 {
  // CHECK-LABEL: func @eq
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %equal = arith.cmpi eq, %self, %other : i32
    return %equal : i1
  }

  // CHECK-LABEL: func @neq
  func.func @neq(%self: i32, %other: i32) -> i1 {
    %equal = trait.method.call @PartialEq::@eq<i32>(%self, %other): (!trait.self, !trait.self) -> i1 to (i32,i32) -> i1
    %true = arith.constant 1 : i1
    %not_equal = arith.xori %equal, %true : i1
    return %not_equal : i1
  }
}


// CHECK-LABEL: func @foo
!T = !trait.poly<0,[@PartialEq]>
func.func @foo(%x: !T, %y: !T) -> i1 {
  // CHECK: %[[EQUAL:.*]] = trait.method.call @PartialEq
  %equal = trait.method.call @PartialEq::@eq<!T>(%x, %y) : (!trait.self, !trait.self) -> i1 to (!T,!T) -> i1
  return %equal : i1
}

// CHECK-LABEL: func @bar
func.func @bar(%x: i32, %y: i32) -> i1 {
  // CHECK: %[[EQUAL:.*]] = trait.func.call @foo
  %equal = trait.func.call @foo(%x, %y) : (!T,!T) -> i1 to (i32,i32) -> i1
  return %equal : i1
}
