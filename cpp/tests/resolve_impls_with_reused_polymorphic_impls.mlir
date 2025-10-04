// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

!A = !trait.poly<0>
trait.trait @A[!A] {
  func.func @a() {
    return
  }
}

trait.impl @A_impl for @A[i1] {}

!B = !trait.poly<1>
trait.trait @B[!B] {}

// 0-tuple impl for @B
trait.impl @B_tuple_impl_arity_0 for @B[tuple<>] {}

// 1-tuple impl for @B
!C = !trait.poly<2>
trait.impl @B_tuple_impl_arity_1 for @B[tuple<!C>] where [
  @A[!C]
] {}

// 2-tuple impl for @B
!D = !trait.poly<3>
!E = !trait.poly<4>
trait.impl @B_tuple_impl_arity_2 for @B[tuple<!D, !E>] where [
  @A[!D],
  @A[!E]
] {}

// polymorphic impl for @A
!F = !trait.poly<5>
trait.impl @A_polymorphic_impl for @A[!F] where [
  @B[!F]
] {
  func.func @a() {
    return
  }
}

func.func @main() {
  // CHECK: trait.witness @"A_polymorphic_impl_tuple<tuple<>, tuple<i1>>_p"
  %0 = trait.allege @A[tuple<tuple<>, tuple<i1>>]
  trait.method.call %0 @A[tuple<tuple<>, tuple<i1>>]::@a() : () -> ()
  return
}

// CHECK: trait.proof @"A_polymorphic_impl_tuple<>_p"
// CHECK: trait.proof @B_tuple_impl_arity_1_i1_p
// CHECK: trait.proof @"A_polymorphic_impl_tuple<i1>_p"
// CHECK: trait.proof @"B_tuple_impl_arity_2_tuple<>_tuple<i1>_p"
// CHECK: trait.proof @"A_polymorphic_impl_tuple<tuple<>, tuple<i1>>_p"
