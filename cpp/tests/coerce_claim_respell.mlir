// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A claim-respell coerce: the operand is an application claim, and the result is
// the SAME application with one argument respelled through a cited child
// equality. This is the shape a bridge cast produces when it respells a claim's
// trait application. The proof backing the operand claim is PRESERVED on the
// result (a respell compares modulo the receipt, never exchanges it), so the
// deep no-swap clause is satisfied and the congruence closure lifts the child
// equality through the application argument.

!S = !trait.poly<0>

trait.trait @Conv[!S] {
  trait.assoc_type @At
}

trait.trait @Safe[!trait.poly<0>, !trait.poly<1>] {
}

trait.impl @Conv_i1 for @Conv[i1] {
  trait.assoc_type @At = i64
}

trait.impl @Safe_impl for @Safe[i32, i64] {
}

trait.proof @Safe_proof proves @Safe_impl for @Safe[i32, i64] given []

// CHECK-LABEL: func.func @respell_preserves_receipt
// CHECK: trait.coerce %{{.*}} : !trait.claim<@Safe[i32, i64] by @Safe_proof> to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof>
func.func @respell_preserves_receipt()
    -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof> {
  %safe = trait.witness @Safe_proof for @Safe[i32, i64]
  %eq = trait.witness proj_resolve !trait.proj<@Conv[i1], "At"> resolves i64 by @Conv_i1
    : !trait.claim<!trait.proj<@Conv[i1], "At"> = i64>
  %c = trait.coerce %safe : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof>
    via (%eq) : (!trait.claim<!trait.proj<@Conv[i1], "At"> = i64>)
  return %c : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">] by @Safe_proof>
}
