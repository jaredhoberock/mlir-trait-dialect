// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

!S = !trait.poly<0>
!O = !trait.poly<1>
// CHECK-NOT: @PartialEq
trait.trait @PartialEq [!S,!O] {
  func.func private @eq(!S, !O) -> i1

  func.func @neq(%self: !S, %other: !O) -> i1 {
    %p = trait.assume @PartialEq[!S,!O]
    %equal = trait.method.call @PartialEq::@eq<%p>(%self, %other)
      : (!S, !O) -> i1
      as !trait.claim<@PartialEq[!S,!O]> (!S, !O) -> i1
    %true = arith.constant 1 : i1
    %res = arith.xori %equal, %true : i1
    return %res : i1
  }
}

// CHECK-NOT: @PartialEq
trait.impl for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

!T = !trait.poly<2>

// CHECK-LABEL: func.func @foo_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @__trait_PartialEq_impl_i32_i32_eq
func.func @foo(%p: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %res = trait.method.call @PartialEq::@eq<%p>(%x, %y)
    : (!S,!O) -> i1
    as !trait.claim<@PartialEq[!T,!T]> (!T,!T) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @bar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @foo_i32
func.func @bar(%x: i32, %y: i32) -> i1 {
  %p = trait.witness @PartialEq[i32,i32]
  %res = trait.func.call @foo(%p, %x, %y)
    : (!trait.claim<@PartialEq[!T,!T]>, !T, !T) -> i1
    as (!trait.claim<@PartialEq[i32,i32]>, i32, i32) -> i1
  return %res : i1
}

// CHECK-LABEL: func.func @baz_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @__trait_PartialEq_impl_i32_i32_eq
// CHECK: call @__trait_PartialEq_impl_i32_i32_neq
func.func @baz(%p: !trait.claim<@PartialEq[!T,!T]>, %x: !T, %y: !T) -> i1 {
  %eq = trait.method.call @PartialEq::@eq<%p>(%x, %y)
    : (!S,!O) -> i1
    as !trait.claim<@PartialEq[!T,!T]> (!T,!T) -> i1

  %neq = trait.method.call @PartialEq::@neq<%p>(%x, %y)
    : (!S,!O) -> i1
    as !trait.claim<@PartialEq[!T,!T]> (!T,!T) -> i1

  %res = arith.ori %eq, %neq : i1
  return %res : i1
}

// CHECK-LABEL: func.func @qux
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: call @baz_i32
func.func @qux(%x: i32, %y: i32) -> i1 {
  %p = trait.witness @PartialEq[i32,i32]
  %result = trait.func.call @baz(%p, %x, %y)
    : (!trait.claim<@PartialEq[!T,!T]>, !T,!T) -> i1
    as (!trait.claim<@PartialEq[i32,i32]>, i32,i32) -> i1
  return %result : i1
}
