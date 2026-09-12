// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An op reads the evidence it holds children first, so a given operand that is
// itself a derive from a projection-spelling header is read through the rules
// that derive's own givens contribute. @Idx_blanket spells its index and its
// element through @Ten[Self], and only the binding %view carries reduces those
// to what %idx spells. Read without it, @Idx_blanket's header stands as
// written, and the two consumers below -- a derive holding the @Idx claim as a
// given, and a method call whose receiver it is -- refuse a claim their own
// operand already carries.

!T = !trait.poly<0>
trait.trait private @Ten[!T] {
  trait.assoc_type @Shape
  trait.assoc_type @Element
}

// The box's element forwards to its base's.
!B = !trait.poly<1>
trait.impl private @Ten_box for @Ten[tuple<!B>] where [@Ten[!B]] {
  trait.assoc_type @Shape = i64
  trait.assoc_type @Element = !trait.proj<@Ten[!B], "Element">
}

!S = !trait.poly<2>
!I = !trait.poly<3>
!E = !trait.poly<4>
trait.trait private @Idx[!S, !I, !E] {
  func.func private @at(!S, !I) -> !E
}

!U = !trait.poly<5>
trait.impl private @Idx_blanket
    for @Idx[!U, !trait.proj<@Ten[!U], "Shape">, !trait.proj<@Ten[!U], "Element">]
    where [@Ten[!U]] {
  func.func @at(%self: !U, %i: !trait.proj<@Ten[!U], "Shape">)
      -> !trait.proj<@Ten[!U], "Element"> {
    %r = ub.poison : !trait.proj<@Ten[!U], "Element">
    return %r : !trait.proj<@Ten[!U], "Element">
  }
}

!V = !trait.poly<6>
trait.trait private @Walk[!V] {
}

trait.impl private @Walk_blanket for @Walk[!V]
    where [@Idx[!V, i64, !trait.proj<@Ten[!V], "Element">]] {
}

// CHECK-LABEL: func.func @reads_a_given_derive
// CHECK: trait.derive @Walk[tuple<!trait.poly<7>>] from @Walk_blanket
// CHECK: trait.method.call
!W = !trait.poly<7>
func.func @reads_a_given_derive(%ten: !trait.claim<@Ten[!W]>,
                                %self: tuple<!W>, %i: i64)
    -> !trait.proj<@Ten[tuple<!W>], "Element"> {
  %view = trait.derive @Ten[tuple<!W>] from @Ten_box given(%ten)
    : (!trait.claim<@Ten[!W]>)
  %idx = trait.derive
    @Idx[tuple<!W>, i64, !trait.proj<@Ten[tuple<!W>], "Element">]
    from @Idx_blanket given(%view) : (!trait.claim<@Ten[tuple<!W>]>)
  %walk = trait.derive @Walk[tuple<!W>] from @Walk_blanket given(%idx)
    : (!trait.claim<@Idx[tuple<!W>, i64, !trait.proj<@Ten[tuple<!W>], "Element">]>)
  %e = trait.method.call %idx
    @Idx[tuple<!W>, i64, !trait.proj<@Ten[tuple<!W>], "Element">]::@at(%self, %i)
    : (tuple<!W>, i64) -> !trait.proj<@Ten[tuple<!W>], "Element">
  return %e : !trait.proj<@Ten[tuple<!W>], "Element">
}
