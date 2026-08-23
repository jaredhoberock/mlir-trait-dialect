// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An impl that forwards its associated type through its own type parameter
// gives one type two spellings: the blanket below spells the element through the
// view (`Ten[tuple<!W>]::Element`) and the derive's demand spells it through the
// base (`Ten[!W]::Element`). Head matching compares the two after normalizing
// both, so the derive selects. Compared as written the heads agree and the
// comparison recurses into their arguments, where it equates the view `tuple<!W>`
// with its base `!W` -- an infinite type the occurs check refuses, reported at
// the derive as a recursive substitution.

!V = !trait.poly<0>
trait.trait @Ten[!V] {
  trait.assoc_type @Element
}

trait.impl @Ten_base for @Ten[i64] {
  trait.assoc_type @Element = i32
}

// The view's element forwards to its base's.
!B = !trait.poly<1>
trait.impl @Ten_view for @Ten[tuple<!B>] where [@Ten[!B]] {
  trait.assoc_type @Element = !trait.proj<@Ten[!B], "Element">
}

!S = !trait.poly<2>
!E = !trait.poly<3>
trait.trait @Get[!S, !E] {
}

!T = !trait.poly<4>
trait.impl @Get_blanket for @Get[!T, !trait.proj<@Ten[!T], "Element">]
    where [@Ten[!T]] {
}

// CHECK-LABEL: func.func @forwarded_element
// CHECK: trait.derive @Get[tuple<!trait.poly<5>>, !trait.proj<@Ten[!trait.poly<5>], "Element">] from @Get_blanket
!W = !trait.poly<5>
func.func @forwarded_element(%ten: !trait.claim<@Ten[!W]>) {
  %view = trait.derive @Ten[tuple<!W>] from @Ten_view given(%ten)
    : (!trait.claim<@Ten[!W]>)
  %get = trait.derive @Get[tuple<!W>, !trait.proj<@Ten[!W], "Element">]
    from @Get_blanket given(%view) : (!trait.claim<@Ten[tuple<!W>]>)
  return
}
