// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The template's variable X carries the label of the impl's associated-type
// parameter W. The impl's header parameter U takes X, and the projection
// supplies i1 for W, so the binding tuple<U, W> read at both lists denotes
// tuple<X, i1>. Both lists stand in one substitution, so the argument X is not
// read as an occurrence of W, and the witness certifying tuple<X, i1> verifies.

!X = !trait.poly<4>
!U = !trait.poly<2>
!W = !trait.poly<4>

trait.trait private @Has[!trait.poly<1>] {
  trait.assoc_type @A<[!trait.poly<3>]>
}
trait.impl private @Has_tuple for @Has[tuple<!U>] {
  trait.assoc_type @A<[!W]> = tuple<!U, !W>
}

func.func private @k(%x: !X, %v: tuple<!X, i1>) -> !trait.proj<@Has[tuple<!X>], "A", [i1]> {
  %e = trait.witness proj_resolve !trait.proj<@Has[tuple<!X>], "A", [i1]> resolves tuple<!X, i1> by @Has_tuple[!U = !X]
    : !trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<!X, i1>>
  %p = trait.coerce %v : tuple<!X, i1> to !trait.proj<@Has[tuple<!X>], "A", [i1]> via (%e)
    : (!trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<!X, i1>>)
  return %p : !trait.proj<@Has[tuple<!X>], "A", [i1]>
}
