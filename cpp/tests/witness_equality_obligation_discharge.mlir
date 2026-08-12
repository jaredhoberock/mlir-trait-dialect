// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// The positive counterpart of witness_conditional_impl_undischarged_assumption:
// the witness over the conditional impl @Has_tuple supplies the @X[i32]
// application premise that discharges the impl's assumption, so
// obligation-aware verification accepts it. The premise rides the witness as a
// claim operand beside the certificate.

!U = !trait.poly<0>

trait.trait @X[!U] {}

trait.trait @Has[!U] {
  trait.assoc_type @Out
}

trait.impl @X_i32 for @X[i32] {}

trait.impl @Has_tuple for @Has[tuple<!U>] where [@X[!U]] {
  trait.assoc_type @Out = i64
}

// CHECK-LABEL: func.func @f
// CHECK: trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple given
// CHECK: trait.coerce
func.func @f(%v: !trait.proj<@Has[tuple<i32>], "Out">, %x: !trait.claim<@X[i32]>) -> i64 {
  %eq = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves i64 by @Has_tuple given(%x)
    : (!trait.claim<@X[i32]>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>
  %c = trait.coerce %v : !trait.proj<@Has[tuple<i32>], "Out"> to i64 via (%eq)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = i64>)
  return %c : i64
}
