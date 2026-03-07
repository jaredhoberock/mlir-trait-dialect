// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Test trait.proj.cast: concrete -> projection -> concrete roundtrip.

!T = !trait.poly<0>

trait.trait @Base[!T] {
  trait.assoc_type @Assoc
}

trait.impl @Base_i64 for @Base[i64] {
  trait.assoc_type @Assoc = i1
}

// CHECK-LABEL: func.func @cast_roundtrip
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: !trait.proj
// CHECK: return %{{.*}} : i1
func.func @cast_roundtrip() -> i1 {
  %v = arith.constant true
  %w = trait.witness @Base_i64 for @Base[i64]
  // cast concrete i1 up to projection type
  %up = trait.proj.cast %v, %w : i1 to !trait.proj<@Base[i64], "Assoc" by @Base_i64>
  // cast projection type back down to concrete i1
  %down = trait.proj.cast %up, %w : !trait.proj<@Base[i64], "Assoc" by @Base_i64> to i1
  return %down : i1
}
