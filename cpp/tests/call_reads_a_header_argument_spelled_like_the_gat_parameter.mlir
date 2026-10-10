// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The caller's variable X carries the label of the impl's associated-type
// parameter W. At T := X the callee's operand @Has[tuple<T>]::A<V> reads the
// binding tuple<U, W> through one substitution carrying U to X and W to V, so
// it denotes tuple<X, V>: the caller's coercion of its tuple<X, i1> to
// @Has[tuple<X>]::A<i1> cites that binding, and the call fills V with i1 off
// the projection's own argument.

!X = !trait.poly<4>
!T = !trait.poly<10>
!V = !trait.poly<11>
!U = !trait.poly<2>
!W = !trait.poly<4>

trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) {
  trait.assoc_type @A<[!trait.poly<1>]>
}
trait.impl private @Has_tuple(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>) {
  trait.assoc_type @A<[!trait.poly<1>]> = tuple<!trait.poly<0>, !trait.poly<1>>
}

func.func private @g(%x: !T,
                     %v: !trait.proj<@Has[tuple<!T>], "A", [!V]>,
                     %c: !trait.claim<@Has[tuple<!T>]>) -> !T {
  return %x : !T
}

func.func private @k(%x: !X, %v: tuple<!X, i1>) -> !X {
  %c = trait.derive @Has[tuple<!X>] from @Has_tuple[!trait.poly<4>] given()
  %a = trait.witness proj_resolve !trait.proj<@Has[tuple<!X>], "A", [i1]> resolves tuple<!X, i1> by @Has_tuple[!trait.poly<4>]
    : !trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<!X, i1>>
  %w = trait.coerce %v : tuple<!X, i1> to !trait.proj<@Has[tuple<!X>], "A", [i1]> via (%a)
    : (!trait.claim<!trait.proj<@Has[tuple<!X>], "A", [i1]> = tuple<!X, i1>>)
  %r = trait.func.call @g(%x, %w, %c)
    : (!X, !trait.proj<@Has[tuple<!X>], "A", [i1]>, !trait.claim<@Has[tuple<!X>]>) -> !X
  return %r : !X
}

// Both clones are monomorphic: X := i64 from main's operand, and the callee's
// operand reduces to tuple<i64, i1> rather than to tuple<i64, i64>.
// CHECK: func.func {{.*}}@g_{{[a-z0-9]+}}(%{{.*}}: i64, %{{.*}}: tuple<i64, i1>) -> i64
// CHECK: func.func {{.*}}@k_{{[a-z0-9]+}}(%{{.*}}: i64, %{{.*}}: tuple<i64, i1>) -> i64
// CHECK: call @g_
func.func @main(%a: i64, %b: tuple<i64, i1>) -> i64 {
  %r = trait.func.call @k(%a, %b) : (i64, tuple<i64, i1>) -> i64
  return %r : i64
}
