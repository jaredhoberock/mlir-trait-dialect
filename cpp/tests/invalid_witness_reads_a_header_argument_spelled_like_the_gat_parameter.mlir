// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The template's variable X carries the label of the impl's associated-type
// parameter W. tuple<i1, i1> is what the binding tuple<U, W> would denote if
// the argument X, stamped in for U, were then read as an occurrence of W. One
// substitution over both lists denotes tuple<X, i1> instead, and the witness
// certifying tuple<i1, i1> is refused.

!X = !trait.poly<4>
!U = !trait.poly<2>
!W = !trait.poly<4>

trait.trait private @Has[!trait.poly<1>] {
  trait.assoc_type @A<[!trait.poly<3>]>
}
trait.impl private @Has_tuple for @Has[tuple<!U>] {
  trait.assoc_type @A<[!W]> = tuple<!U, !W>
}

func.func private @k(%x: !X, %v: tuple<i1, i1>) -> !trait.proj<@Has[tuple<!X>], "A", [i1]> {
  // expected-error @below {{impl '@Has_tuple' binds the projection to 'tuple<!trait.poly<4>, i1>', not the certified resolution 'tuple<i1, i1>'}}
  %e = trait.witness proj_resolve !trait.proj<@Has[tuple<!X>], "A", [i1]> resolves tuple<i1, i1> by @Has_tuple[!U = !X]
    : !trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<i1, i1>>
  %p = trait.coerce %v : tuple<i1, i1> to !trait.proj<@Has[tuple<!X>], "A", [i1]> via (%e)
    : (!trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<i1, i1>>)
  return %p : !trait.proj<@Has[tuple<!X>], "A", [i1]>
}
