// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" | FileCheck %s

// A bare-alias marked coerce discharges when both lookups ground to the SAME
// type. Monomorphization resolves @Base[i64]::A and @Base[i64]::B, both bound to
// i1 by the impl; the two endpoints respell to i1, the coerce becomes reflexive,
// and the folder collapses it -- the promise the alias made is kept, and the op
// dissolves.

trait.trait @Base[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.impl @Base_i64 for @Base[i64] {
  trait.assoc_type @A = i1
  trait.assoc_type @B = i1
}

// CHECK-LABEL: func.func @use
// CHECK-SAME: (%[[A:.*]]: i1) -> i1
// CHECK-NOT: trait.coerce
// CHECK-NOT: !trait.proj
// CHECK: return %[[A]] : i1
func.func @use(%x: !trait.proj<@Base[i64], "A">) -> !trait.proj<@Base[i64], "B"> {
  %y = trait.coerce %x : !trait.proj<@Base[i64], "A">
    to !trait.proj<@Base[i64], "B"> unproven
  return %y : !trait.proj<@Base[i64], "B">
}
