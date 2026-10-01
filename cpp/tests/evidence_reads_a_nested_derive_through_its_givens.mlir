// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// An op reads the evidence it holds children first, so a receiver that is a
// derive from a projection-spelling header is read through the rules that
// derive's own givens contribute. @Idx_blanket spells its index and its element
// through @Ten[Self], and only the binding %view carries reduces the index to
// the i64 the method call below passes. Read without it, @Idx_blanket's header
// stands as written, and the call refuses an index its receiver's own evidence
// already settles.

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
  trait.method @at(!S, !I) -> !E
}

!U = !trait.poly<5>
trait.impl private @Idx_blanket
    for @Idx[!U, !trait.proj<@Ten[!U], "Shape">, !trait.proj<@Ten[!U], "Element">]
    where [@Ten[!U]] {
  trait.method @at(%self: !U, %i: !trait.proj<@Ten[!U], "Shape">)
      -> !trait.proj<@Ten[!U], "Element"> {
    %r = ub.poison : !trait.proj<@Ten[!U], "Element">
    trait.return %r : !trait.proj<@Ten[!U], "Element">
  }
}

// CHECK-LABEL: func.func @reads_a_given_derive
// CHECK: trait.method.call
!W = !trait.poly<7>
func.func @reads_a_given_derive(%ten: !trait.claim<@Ten[!W]>,
                                %self: tuple<!W>, %i: i64)
    -> !trait.proj<@Ten[tuple<!W>], "Element"> {
  %view = trait.derive @Ten[tuple<!W>] from @Ten_box[!B = !W] given(%ten)
    : (!trait.claim<@Ten[!W]>)
  %idx = trait.derive
    @Idx[tuple<!W>, !trait.proj<@Ten[tuple<!W>], "Shape">,
         !trait.proj<@Ten[tuple<!W>], "Element">]
    from @Idx_blanket[!U = tuple<!W>] given(%view)
    : (!trait.claim<@Ten[tuple<!W>]>)
  %e = trait.method.call %idx
    @Idx[tuple<!W>, !trait.proj<@Ten[tuple<!W>], "Shape">,
         !trait.proj<@Ten[tuple<!W>], "Element">]::@at(%self, %i)
    : (tuple<!W>, i64) -> !trait.proj<@Ten[tuple<!W>], "Element">
  return %e : !trait.proj<@Ten[tuple<!W>], "Element">
}
