// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWER

// The call-boundary accept path through a CONDITIONAL impl. The supplied
// equality witness cites @Has_tuple, whose assumption @X[!U] is met for
// !U = i32 by @X_i32, and it carries the @X[i32] claim as the premise that
// discharges that assumption for the obligation-aware audit. The equality claim
// crosses the call and survives to the leftover check as a call operand; there
// its projection ground-resolves to i64 through the discharged conditional
// selection, the endpoints meet, and the claim is accepted. Without @X_i32 the
// projection would not resolve and the claim would stay a leftover
// (witness_conditional_impl_undischarged_assumption.mlir refuses the premise-less
// witness outright).

!S = !trait.poly<0>
!T = !trait.poly<1>
!U = !trait.poly<2>

trait.trait @X[!U] {}

trait.trait @Has[!S] {
  trait.assoc_type @Out
}

trait.impl @X_i32 for @X[i32] {}

trait.impl @Has_tuple for @Has[tuple<!U>] where [@X[!U]] {
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
func.func @main(%pv: !trait.proj<@Has[tuple<i32>], "Out">) -> i64 {
  %w = trait.witness @X_i32 for @X[i32]
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple given(%w)
    : (!trait.claim<@X[i32] by @X_i32>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  // LOWER: call @gen{{.*}}(%arg0) : (i64) -> i64
  %r = trait.func.call @gen(%pv, %eq)
    : (!trait.proj<@Has[tuple<i32>], "Out">, !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>) -> i64
  return %r : i64
}
