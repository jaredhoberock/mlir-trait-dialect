// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWER

// A generic callee whose equality formal is projection-headed over its own type
// parameter. The caller supplies the callee's declared predicate specialized at
// the call -- an impl-cited equality witness -- as the trailing claim argument.
//
// An equality-arm operand carries its evidence as the value itself, so the call
// lowers without waiting for a proof the arm never carries. The callee
// specializes at !S = i32, !T = i64; the projection resolves to i64, the coerce
// discharges, and the equality claim parameter shrinks 1:0 at the barrier. The
// specialized instance is monomorphic with no residual claim, and the whole
// module lowers clean.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @Has[!S] {
  trait.assoc_type @Out
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @gen
// CHECK: trait.coerce
func.func @gen(%v: !trait.proj<@Has[!S], "Out">, %c: !trait.claim<!trait.proj<@Has[!S], "Out"> = !T>) -> !T {
  %r = trait.coerce %v : !trait.proj<@Has[!S], "Out"> to !T via (%c)
    : (!trait.claim<!trait.proj<@Has[!S], "Out"> = !T>)
  return %r : !T
}

// LOWER: func.func @gen{{.*}}(%arg0: i64) -> i64
// LOWER-NOT: trait.claim
// LOWER: return %arg0 : i64
func.func @main(%pv: !trait.proj<@Has[i32], "Out">) -> i64 {
  %eq = trait.witness proj_resolve !trait.proj<@Has[i32], "Out"> resolves i64 by @Has_i32
    : !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>
  // LOWER: call @gen{{.*}}(%arg0) : (i64) -> i64
  %r = trait.func.call @gen(%pv, %eq)
    : (!trait.proj<@Has[i32], "Out">, !trait.claim<!trait.proj<@Has[i32], "Out"> = i64>) -> i64
  return %r : i64
}
