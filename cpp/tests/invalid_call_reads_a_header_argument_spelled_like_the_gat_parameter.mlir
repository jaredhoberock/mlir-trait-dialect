// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The caller's variable X carries the label of the impl's associated-type
// parameter W. @Has[tuple<X>]::A<i1> denotes tuple<X, i1>, so a witness that
// it resolves to tuple<i1, i1> is refused. tuple<i1, i1> is what the binding
// would denote if the argument X, stamped in for the header parameter U, were
// then read as an occurrence of W.

// @k labels its parameters by position, so its first, Y, takes label 0 and X
// takes label 1, the GAT parameter's.
!Y = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<10>
!V = !trait.poly<11>
!U = !trait.poly<2>
!W = !trait.poly<1>

trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]>
}
trait.impl private @Has_tuple(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = tuple<!trait.poly<0>, !trait.poly<1>>
}

func.func private @g(%x: !trait.poly<0>,
                     %v: !trait.proj<@Has[tuple<!trait.poly<0>>], "A", [!trait.poly<1>]>,
                     %c: !trait.claim<@Has[tuple<!trait.poly<0>>]>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

func.func private @k(%y: !Y, %x: !X, %v: tuple<i1, i1>) -> !X {
  %c = trait.derive @Has[tuple<!X>] from @Has_tuple[!trait.poly<1>] given()
  // expected-error @below {{impl '@Has_tuple' binds the projection to 'tuple<!trait.poly<1>, i1>', not the certified resolution 'tuple<i1, i1>'}}
  %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!X>], "A", [i1]> resolves tuple<i1, i1> by @Has_tuple[!trait.poly<1>]
    : !trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<i1, i1>>
  %w = trait.coerce %v : tuple<i1, i1> to !trait.proj<@Has[tuple<!X>], "A", [i1]> via (%a)
    : (!trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<i1, i1>>)
  %r = trait.func.call @g(%x, %w, %c)
    : (!X, !trait.proj<@Has[tuple<!X>], "A", [i1]>, !trait.claim<@Has[tuple<!X>]>) -> !X
  return %r : !X
}

func.func @main(%a: i64, %b: tuple<i1, i1>) -> i64 {
  %r = trait.func.call @k(%a, %a, %b) : (i64, i64, tuple<i1, i1>) -> i64
  return %r : i64
}
