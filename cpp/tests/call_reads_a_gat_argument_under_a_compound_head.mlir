// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A callee variable that stands only inside the associated-type arguments of a
// projection whose head is a compound type served by a conditional impl. At the
// call the head is ground (tuple<i64>) while the argument position still spells
// the variable, so the reading fills it only after rebuilding the declaration at
// what it has read and reducing that projection -- which the head's own spelling
// determines even where its associated-type arguments hold a variable.

!T = !trait.poly<10>
!V = !trait.poly<11>
!U = !trait.poly<2>

trait.trait private @M0[!trait.poly<0>] {}
trait.impl private @M0_i64 for @M0[i64] {}

trait.trait private @Has[!trait.poly<1>] {
  trait.assoc_type @A<[!trait.poly<3>]>
}
trait.impl private @Has_tuple for @Has[tuple<!U>] where [@M0[!U]] {
  trait.assoc_type @A<[!trait.poly<4>]> = !trait.poly<4>
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
  %e = trait.witness proj_resolve !trait.proj<@Has[tuple<i64>], "A", [i1]> resolves i1 by @Has_tuple[!U = i64] given(%w)
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
