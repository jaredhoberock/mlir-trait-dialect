// RUN: opt --convert-to-llvm %s | FileCheck %s

// CHECK-NOT: @PartialEq
trait.trait @PartialEq {
  func.func private @eq(!trait.self, !trait.self) -> i1

  func.func @neq(%self: !trait.self, %other: !trait.self) -> i1 {
    %equal = trait.method.call @PartialEq<!trait.self>::@eq(%self, %other) : (!trait.self, !trait.self) -> i1
    %true = arith.constant 1 : i1
    %result = arith.xori %equal, %true : i1
    return %result : i1
  }
}

// CHECK-NOT: @PartialEq<i32>
trait.impl @PartialEq<i32> {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %result = arith.cmpi eq, %self, %other : i32
    return %result : i1
  }
}

!T = !trait.poly<0,[@PartialEq]>

// CHECK-LABEL: llvm.func @foo_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @__trait_PartialEq_impl_i32_eq
func.func @foo(%x: !T, %y: !T) -> i1 {
  %result = trait.method.call @PartialEq<!T>::@eq(%x, %y) : (!T,!T) -> i1
  return %result : i1
}

// CHECK-LABEL: llvm.func @bar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @foo_i32
func.func @bar(%x: i32, %y: i32) -> i1 {
  %result = trait.func.call @foo(%x, %y) : (i32,i32) -> i1
  return %result : i1
}

// CHECK-LABEL: llvm.func @baz_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @__trait_PartialEq_impl_i32_eq
// CHECK: llvm.call @__trait_PartialEq_impl_i32_neq
func.func @baz(%x: !T, %y: !T) -> i1 {
  %eq = trait.method.call @PartialEq<!T>::@eq(%x, %y) : (!T,!T) -> i1
  %neq = trait.method.call @PartialEq<!T>::@neq(%x, %y) : (!T,!T) -> i1
  %result = arith.ori %eq, %neq : i1
  return %result : i1
}

// CHECK-LABEL: llvm.func @qux
// CHECK-NOTE: builtin.unrealized_conversion_cast
// CHECK: llvm.call @baz_i32
func.func @qux(%x: i32, %y: i32) -> i1 {
  %result = trait.func.call @baz(%x, %y) : (i32,i32) -> i1
  return %result : i1
}
