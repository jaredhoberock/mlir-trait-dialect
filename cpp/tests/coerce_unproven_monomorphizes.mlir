// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" | FileCheck %s

// The discharge a marked coerce rides. Its endpoints reconcile a projection
// against the ground type an impl supplies; once monomorphization resolves the
// ground projection through that impl's binding, the projection respells to i1,
// the coerce becomes reflexive, and the folder collapses it -- the op mints
// nothing and dissolves, leaving a straight-through function.

trait.trait @Base[!trait.poly<0>] {
  trait.assoc_type @Assoc
}

trait.impl @Base_i64 for @Base[i64] {
  trait.assoc_type @Assoc = i1
}

// CHECK-LABEL: func.func @use
// CHECK-SAME: (%[[A:.*]]: i1) -> i1
// CHECK-NOT: trait.coerce
// CHECK-NOT: !trait.proj
// CHECK: return %[[A]] : i1
func.func @use(%x: !trait.proj<@Base[i64], "Assoc">) -> i1 {
  %y = trait.coerce %x : !trait.proj<@Base[i64], "Assoc"> to i1 unproven
  return %y : i1
}
