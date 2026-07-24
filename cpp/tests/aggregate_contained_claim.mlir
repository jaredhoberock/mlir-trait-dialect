// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Records that verification does not look inside an aggregate for a
// !trait.claim. The cast below produces a tuple whose first element is a
// monomorphic, unproven @U[i32] claim; the op verifier accepts the result type
// without inspecting the tuple. The claim gates elsewhere in the pipeline that
// scan op results -- the post-monomorphization leftover check for unproven
// monomorphic claims -- key on the root type of each result, so a claim behind a
// tuple constructor is not seen there either. verifyMonomorphs walks a function
// signature deeply, but only a function signature; an aggregate claim in an op
// result is outside its reach. Pinned as accepted so a rule that rejects
// aggregate-contained claims registers as a change here.

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}
trait.impl @Fold_i32 for @Fold[i32] {
  trait.assoc_type @Item = i64
}
trait.trait @U[!trait.poly<0>] {}

// CHECK-LABEL: func.func @aggregate_result_claim
// CHECK: trait.proj.cast
// CHECK-SAME: tuple<!trait.claim<@U[i32]>, i32>
func.func @aggregate_result_claim(
    %v: tuple<!trait.proj<@Fold[i32], "Item">, i32>,
    %c: !trait.claim<@Fold[i32]>) {
  %cast = trait.proj.cast %v, %c
    : tuple<!trait.proj<@Fold[i32], "Item">, i32>
    to tuple<!trait.claim<@U[i32]>, i32>
    by !trait.claim<@Fold[i32]>
  return
}
