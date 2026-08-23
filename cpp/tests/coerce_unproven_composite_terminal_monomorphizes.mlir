// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" | FileCheck %s

// A marked coerce whose binding terminal still carries a projection discharges
// like any other: monomorphization grounds BOTH lookups at once. @Base[i64]::A
// is bound to tuple<i1>, and @Base[i64]::B grounds to i1 in the same step, so
// the two endpoints respell to tuple<i1>, the coerce becomes reflexive, and the
// folder collapses it. This is a bare projection aliased to a composite that
// still carries a projection -- one whole lookup standing for a spelling that
// carries a second lookup, both grounding at one monomorphization.

trait.trait @Base[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.impl @Base_i64 for @Base[i64] {
  trait.assoc_type @A = tuple<i1>
  trait.assoc_type @B = i1
}

// CHECK-LABEL: func.func @use
// CHECK-SAME: (%[[A:.*]]: tuple<i1>) -> tuple<i1>
// CHECK-NOT: trait.coerce
// CHECK-NOT: !trait.proj
// CHECK: return %[[A]] : tuple<i1>
func.func @use(%x: !trait.proj<@Base[i64], "A">)
    -> tuple<!trait.proj<@Base[i64], "B">> {
  %y = trait.coerce %x : !trait.proj<@Base[i64], "A">
    to tuple<!trait.proj<@Base[i64], "B">> unproven
  return %y : tuple<!trait.proj<@Base[i64], "B">>
}
