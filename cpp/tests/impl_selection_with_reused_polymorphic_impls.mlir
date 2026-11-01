// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

!A = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!A]>) members [default @A.a]
func.func private @A.a(%self: !trait.claim<@A[!A]>) {
  return
}

trait.impl private @A_impl(%self: !trait.claim<@A[i1]>) {
}

!B = !trait.poly<1>
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>)

// 0-tuple impl for @B
trait.impl private @B_tuple_impl_arity_0(%self: !trait.claim<@B[tuple<>]>) {
}

// 1-tuple impl for @B
!C = !trait.poly<2>
trait.impl private @B_tuple_impl_arity_1(%self: !trait.claim<@B[tuple<!trait.poly<0>>]>, %a: !trait.claim<@A[!trait.poly<0>]>) {
}

// 2-tuple impl for @B
!D = !trait.poly<3>
!E = !trait.poly<4>
trait.impl private @B_tuple_impl_arity_2(%self: !trait.claim<@B[tuple<!trait.poly<0>, !trait.poly<1>>]>, %a: !trait.claim<@A[!trait.poly<0>]>, %a_1: !trait.claim<@A[!trait.poly<1>]>) {
}

// polymorphic impl for @A
!F = !trait.poly<5>
trait.impl private @A_polymorphic_impl(%self: !trait.claim<@A[!trait.poly<0>]>, %b: !trait.claim<@B[!trait.poly<0>]>) members [@A.a -> @A_polymorphic_impl.a] {
}
func.func private @A_polymorphic_impl.a(%self: !trait.claim<@A[!trait.poly<0>]>, %b: !trait.claim<@B[!trait.poly<0>]>) {
  return
}

func.func @main() {
  // CHECK: trait.witness @A_polymorphic_impl_{{.*}}_p
  %0 = trait.allege @A[tuple<tuple<>, tuple<i1>>]
  trait.func.call @A.a(%0) : (!trait.claim<@A[tuple<tuple<>, tuple<i1>>]>) -> ()
  return
}

// CHECK: trait.proof private @A_polymorphic_impl_{{.*}}_p
// CHECK: trait.proof private @B_tuple_impl_arity_1_{{.*}}_p
// CHECK: trait.proof private @A_polymorphic_impl_{{.*}}_p
// CHECK: trait.proof private @B_tuple_impl_arity_2_{{.*}}_p
// CHECK: trait.proof private @A_polymorphic_impl_{{.*}}_p
