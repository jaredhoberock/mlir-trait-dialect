// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The two spellings of one type need not share a head. The blanket below spells
// the element as a lookup through the container (`Ten[tuple<i64, X>]::Element`)
// while the derive's demand spells it as the element the container was built
// from (`Elem[!W]::E`); one resolution hop through the container's element
// binding joins them. Compared as written the heads simply differ, and the
// derive is refused for a projection mismatch.
//
// The match that resolves the container runs inside the derive's own
// specialization build, over types that build already instantiated, so it also
// pins that the inner build's fresh variables are fresh with respect to them: an
// inner variable that aliased `!W`'s would unify the container's element with a
// lookup that contains it and find no candidate impl at all.

!A = !trait.poly<0>
trait.trait @Elem[!A] {
  trait.assoc_type @E
}

!P = !trait.poly<1>
trait.trait @Ten[!P] {
  trait.assoc_type @Element
}

// The container's element is the second half of the pair it is built from.
!S = !trait.poly<2>
!X = !trait.poly<3>
trait.impl @Ten_pair for @Ten[tuple<!S, !X>] {
  trait.assoc_type @Element = !X
}

!C = !trait.poly<4>
!E = !trait.poly<5>
trait.trait @Get[!C, !E] {
}

!T = !trait.poly<6>
trait.impl @Get_blanket for @Get[!T, !trait.proj<@Ten[!T], "Element">]
    where [@Ten[!T]] {
}

// CHECK-LABEL: func.func @resolved_element
// CHECK: trait.derive @Get[tuple<i64, !trait.proj<@Elem[!trait.poly<7>], "E">>, !trait.proj<@Elem[!trait.poly<7>], "E">] from @Get_blanket
!W = !trait.poly<7>
func.func @resolved_element(
    %ten: !trait.claim<@Ten[tuple<i64, !trait.proj<@Elem[!W], "E">>]>) {
  %get = trait.derive
    @Get[tuple<i64, !trait.proj<@Elem[!W], "E">>, !trait.proj<@Elem[!W], "E">]
    from @Get_blanket given(%ten)
    : (!trait.claim<@Ten[tuple<i64, !trait.proj<@Elem[!W], "E">>]>)
  return
}
