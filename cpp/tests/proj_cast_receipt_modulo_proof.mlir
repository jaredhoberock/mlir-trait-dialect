// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A proven cast reconciles a proven input claim with a result claim that spells
// the same claim through a projection and carries no receipt. Once the
// projection resolves, the two endpoints name one application and differ only
// in the proof receipt on the input -- and a receipt informs the reader but
// never fails a correct cast, so the endpoints compare modulo it and the cast
// verifies. This is the between-state shape the collective machinery emits en
// masse (a proven @ConvergenceSafeIn input against the same claim spelled with a
// @Convergence::ConvergedAt projection), reduced here to a self-contained row.

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

trait.proof @Conv_proof proves @Conv_i1 for @Conv[i1] given []
trait.proof @Safe_proof proves @Safe_impl for @Safe[i32, i64] given []

// CHECK-LABEL: func.func @crumb
// CHECK: trait.proj.cast
// CHECK-SAME: !trait.claim<@Safe[i32, i64] by @Safe_proof>
// CHECK-SAME: to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
func.func @crumb() -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]> {
  %safe = trait.witness @Safe_proof for @Safe[i32, i64]
  %conv = trait.witness @Conv_proof for @Conv[i1]
  %cast = trait.proj.cast %safe, %conv
    : !trait.claim<@Safe[i32, i64] by @Safe_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
    by !trait.claim<@Conv[i1] by @Conv_proof>
  return %cast : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
}
