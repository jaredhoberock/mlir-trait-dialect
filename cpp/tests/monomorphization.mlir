// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!S = !trait.poly<0>
!O = !trait.poly<1>
!W = !trait.witness<@PartialEq[!S,!O]>
// CHECK-NOT: @PartialEq
trait.trait @PartialEq [!S,!O] {
  trait.method @eq(!S, !O) -> i1

  trait.method @neq(%w: !W, %self: !S, %other: !O) -> i1 {
    %equal = trait.method.call @PartialEq::@eq<%w>(%self, %other)
      : (!S, !O) -> i1
      as !W (!S, !O) -> i1
    %true = arith.constant 1 : i1
    %res = arith.xori %equal, %true : i1
    return %res : i1
  }
}

// CHECK-NOT: @PartialEq
trait.impl @PartialEq[i32,i32] {
  trait.method @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

!T = !trait.poly<2>

// CHECK-LABEL: func.func @foo_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @__trait_PartialEq_impl_i32_i32_eq
func.func @foo(%w: !trait.witness<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %res = trait.method.call @PartialEq::@eq<%w>(%x, %y)
    : (!S,!O) -> i1
    as !trait.witness<@PartialEq[!T,!T]> (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @bar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @foo_i32
func.func @bar(%x: i32, %y: i32) -> i1 {
  %w = trait.witness : !trait.witness<@PartialEq[i32,i32]>
  %res = trait.func.call @foo(%w, %x, %y)
    : (!trait.witness<@PartialEq[!T,!T]>, !T, !T) -> i1
    as (!trait.witness<@PartialEq[i32,i32]>, i32, i32) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @baz_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @__trait_PartialEq_impl_i32_i32_eq
// CHECK: call @__trait_PartialEq_impl_i32_i32_neq
func.func @baz(%w: !trait.witness<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %eq = trait.method.call @PartialEq::@eq<%w>(%x, %y)
    : (!S,!O) -> i1
    as !trait.witness<@PartialEq[!T,!T]> (!T,!T) -> i1

  %neq = trait.method.call @PartialEq::@neq<%w>(%w, %x, %y)
    : (!W,!S,!O) -> i1
    as !trait.witness<@PartialEq[!T,!T]> (!trait.witness<@PartialEq[!T,!T]>, !T,!T) -> i1

  %res = arith.ori %eq, %neq : i1
  return %res : i1
}

// CHECK-LABEL: func.func @qux
// CHECK-NOTE: builtin.unrealized_conversion_cast
// CHECK: call @baz_i32
func.func @qux(%x: i32, %y: i32) -> i1 {
  %w = trait.witness : !trait.witness<@PartialEq[i32,i32]>
  %result = trait.func.call @baz(%w, %x, %y)
    : (!trait.witness<@PartialEq[!T,!T]>, !T,!T) -> i1
    as (!trait.witness<@PartialEq[i32,i32]>, i32,i32) -> i1
  return %result : i1
}
