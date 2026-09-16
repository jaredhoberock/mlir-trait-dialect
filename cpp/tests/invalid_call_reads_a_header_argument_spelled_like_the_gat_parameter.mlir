// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The caller's variable X carries the label of the impl's associated-type
// parameter W. At T := X the callee's operand @Has[tuple<T>]::A<V> denotes
// tuple<X, V>, so no argument for V makes it tuple<i1, i1> and the call is
// ill-typed. tuple<i1, i1> is what the binding would denote if the argument X,
// stamped in for the header parameter U, were then read as an occurrence of W.

!X = !trait.poly<4>
!T = !trait.poly<10>
!V = !trait.poly<11>
!U = !trait.poly<2>
!W = !trait.poly<4>

trait.trait private @Has[!trait.poly<1>] {
  trait.assoc_type @A<[!trait.poly<3>]>
}
trait.impl private @Has_tuple for @Has[tuple<!U>] {
  trait.assoc_type @A<[!W]> = tuple<!U, !W>
}
trait.proof private @Has_tuple_p proves @Has_tuple for @Has[tuple<!X>] given []

func.func private @g(%x: !T,
                     %v: !trait.proj<@Has[tuple<!T>], "A", [!V]>,
                     %c: !trait.claim<@Has[tuple<!T>]>) -> !T {
  return %x : !T
}

func.func private @k(%x: !X, %v: tuple<i1, i1>) -> !X {
  %c = trait.witness @Has_tuple_p for @Has[tuple<!X>]
  // expected-error @below {{type mismatch: expected '(!trait.poly<4>, tuple<!trait.poly<4>, i1>, !trait.claim<@Has[tuple<!trait.poly<4>>]>) -> !trait.poly<4>' but found '(!trait.poly<4>, tuple<i1, i1>, !trait.claim<@Has[tuple<!trait.poly<4>>]>) -> !trait.poly<4>'}}
  %r = trait.func.call @g(%x, %v, %c)
    : (!X, tuple<i1, i1>, !trait.claim<@Has[tuple<!X>] by @Has_tuple_p>) -> !X
  return %r : !X
}

func.func @main(%a: i64, %b: tuple<i1, i1>) -> i64 {
  %r = trait.func.call @k(%a, %b) : (i64, tuple<i1, i1>) -> i64
  return %r : i64
}
