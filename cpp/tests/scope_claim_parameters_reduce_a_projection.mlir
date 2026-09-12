// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A declaration's claim parameters are its where clause, and a where clause is
// the parameter environment of everything its body holds. @Give_blanket's
// header spells its second argument as @HasOut[Self]::Out, so the derives below
// -- demanding @Give at i64 -- reach that header only through an equality
// saying the projection IS i64. Neither derive is given that equality: the
// first function carries it as an equality parameter, the second carries an
// application parameter whose trait requires it.

!T = !trait.poly<0>
trait.trait private @HasOut[!T] {
  trait.assoc_type @Out
}

!S = !trait.poly<1>
!O = !trait.poly<2>
trait.trait private @Give[!S, !O] {
}

!U = !trait.poly<3>
trait.impl private @Give_blanket for @Give[!U, !trait.proj<@HasOut[!U], "Out">]
    where [@HasOut[!U]] {
}

// CHECK-LABEL: func.func @from_an_equality_parameter
// CHECK: trait.derive @Give[!trait.poly<4>, i64] from @Give_blanket
!A = !trait.poly<4>
func.func @from_an_equality_parameter(
    %has: !trait.claim<@HasOut[!A]>,
    %eq: !trait.claim<!trait.proj<@HasOut[!A], "Out"> = i64>) {
  %give = trait.derive @Give[!A, i64] from @Give_blanket given(%has)
    : (!trait.claim<@HasOut[!A]>)
  return
}

!B = !trait.poly<5>
trait.trait private @Counted[!B]
    where [@HasOut[!B], !trait.proj<@HasOut[!B], "Out"> = i64] {
}

// CHECK-LABEL: func.func @from_a_trait_requirement
// CHECK: trait.derive @Give[!trait.poly<6>, i64] from @Give_blanket
!C = !trait.poly<6>
func.func @from_a_trait_requirement(
    %counted: !trait.claim<@Counted[!C]>,
    %has: !trait.claim<@HasOut[!C]>) {
  %give = trait.derive @Give[!C, i64] from @Give_blanket given(%has)
    : (!trait.claim<@HasOut[!C]>)
  return
}
