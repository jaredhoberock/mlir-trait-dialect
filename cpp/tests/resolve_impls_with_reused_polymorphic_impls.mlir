// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

!A = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!A]>) {
  trait.method @a() {
    trait.return
  }
}

trait.impl private @A_impl(%self: !trait.claim<@A[i1]>) {}

!B = !trait.poly<1>
trait.trait private @B(%self: !trait.claim<@B[!B]>) {}

// 0-tuple impl for @B
trait.impl private @B_tuple_impl_arity_0(%self: !trait.claim<@B[tuple<>]>) {}

// 1-tuple impl for @B
!C = !trait.poly<2>
trait.impl private @B_tuple_impl_arity_1(%self: !trait.claim<@B[tuple<!C>]>, %a: !trait.claim<@A[!C]>) {}

// 2-tuple impl for @B
!D = !trait.poly<3>
!E = !trait.poly<4>
trait.impl private @B_tuple_impl_arity_2(%self: !trait.claim<@B[tuple<!D, !E>]>, %a: !trait.claim<@A[!D]>, %a_1: !trait.claim<@A[!E]>) {}

// polymorphic impl for @A
!F = !trait.poly<5>
trait.impl private @A_polymorphic_impl(%self: !trait.claim<@A[!F]>, %b: !trait.claim<@B[!F]>) {
  trait.method @a() {
    trait.return
  }
}

func.func @main() {
  // CHECK: trait.witness @A_polymorphic_impl_{{.*}}_p
  %0 = trait.allege @A[tuple<tuple<>, tuple<i1>>]
  trait.method.call %0 @A[tuple<tuple<>, tuple<i1>>]::@a() : () -> ()
  return
}

// CHECK: trait.proof private @A_polymorphic_impl_{{.*}}_p
// CHECK: trait.proof private @B_tuple_impl_arity_1_{{.*}}_p
// CHECK: trait.proof private @A_polymorphic_impl_{{.*}}_p
// CHECK: trait.proof private @B_tuple_impl_arity_2_{{.*}}_p
// CHECK: trait.proof private @A_polymorphic_impl_{{.*}}_p
