// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The caller's variable X carries the label of the impl's associated-type
// parameter W. At T := X the callee's operand @Has[tuple<T>]::A<V> reads the
// binding tuple<U, W> through one substitution carrying U to X and W to V, so
// it denotes tuple<X, V>, and the operand the caller spells fills V with i1.

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

func.func private @k(%x: !X, %v: tuple<!X, i1>) -> !X {
  %c = trait.witness @Has_tuple_p for @Has[tuple<!X>]
  %r = trait.func.call @g(%x, %v, %c)
    : (!X, tuple<!X, i1>, !trait.claim<@Has[tuple<!X>] by @Has_tuple_p>) -> !X
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
