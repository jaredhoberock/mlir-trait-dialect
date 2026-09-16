// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Two impls serve @Has for tuple<U> under disjoint premises, so the module's
// read-only lookup declines the head tuple<i64>: two headers match it. Selection
// has settled that head, and what selection settled is read by the head alone --
// which impl serves a projection is decided by its head application, and the
// impl's binding is a function of the projection's associated-type arguments, so
// the record answers here even though the argument position still spells V.

!T = !trait.poly<10>
!V = !trait.poly<11>
!U = !trait.poly<2>
!U1 = !trait.poly<5>

trait.trait private @M0[!trait.poly<0>] {}
trait.impl private @M0_i64 for @M0[i64] {}
trait.trait private @M1[!trait.poly<6>] {}
trait.impl private @M1_i1 for @M1[i1] {}

trait.trait private @Has[!trait.poly<1>] {
  trait.assoc_type @A<[!trait.poly<3>]>
}
trait.impl private @Has_tuple_m0 for @Has[tuple<!U>] where [@M0[!U]] {
  trait.assoc_type @A<[!trait.poly<4>]> = !trait.poly<4>
}
trait.impl private @Has_tuple_m1 for @Has[tuple<!U1>] where [@M1[!U1]] {
  trait.assoc_type @A<[!trait.poly<7>]> = !trait.poly<7>
}

func.func private @g(%x: !T,
                     %v: !trait.proj<@Has[tuple<!T>], "A", [!V]>,
                     %c: !trait.claim<@M0[!T]>)
    -> !trait.proj<@Has[tuple<!T>], "A", [!V]> {
  return %v : !trait.proj<@Has[tuple<!T>], "A", [!V]>
}

// The clone is monomorphic: T := i64 from the first operand, V := i1 from the
// reduced projection, and the call is a plain func.call to it.
// CHECK: func.func {{.*}}@g_{{[a-z0-9]+}}(%{{.*}}: i64, %{{.*}}: i1
// CHECK: call @g_
func.func @main(%a: i64, %b: i1) -> i1 {
  %w = trait.witness @M0_i64 for @M0[i64]
  %e = trait.witness proj_resolve !trait.proj<@Has[tuple<i64>], "A", [i1]> resolves i1 by @Has_tuple_m0 given(%w)
    : (!trait.claim<@M0[i64] by @M0_i64>)
    : !trait.claim<!trait.proj<@Has[tuple<i64>], "A", [i1]> = i1>
  %p = trait.coerce %b : i1 to !trait.proj<@Has[tuple<i64>], "A", [i1]> via (%e)
    : (!trait.claim<!trait.proj<@Has[tuple<i64>], "A", [i1]> = i1>)
  %r = trait.func.call @g(%a, %p, %w)
    : (i64, !trait.proj<@Has[tuple<i64>], "A", [i1]>, !trait.claim<@M0[i64] by @M0_i64>)
    -> !trait.proj<@Has[tuple<i64>], "A", [i1]>
  %o = trait.coerce %r : !trait.proj<@Has[tuple<i64>], "A", [i1]> to i1 via (%e)
    : (!trait.claim<!trait.proj<@Has[tuple<i64>], "A", [i1]> = i1>)
  return %o : i1
}
