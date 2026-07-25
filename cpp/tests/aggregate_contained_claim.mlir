// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A !trait.claim is SSA evidence that monomorphization erases one-to-zero;
// nested inside an aggregate it has no independent value to erase. The cast
// below produces a tuple whose first element is a monomorphic, unproven @U[i32]
// claim. The verifier scans the input and result types for a claim nested inside
// an aggregate and rejects it, closing the blind spot the result-root claim
// gates elsewhere leave open.

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}
trait.impl @Fold_i32 for @Fold[i32] {
  trait.assoc_type @Item = i64
}
trait.trait @U[!trait.poly<0>] {}

func.func @aggregate_result_claim(
    %v: tuple<!trait.proj<@Fold[i32], "Item">, i32>,
    %c: !trait.claim<@Fold[i32]>) {
  // expected-error @below {{!trait.claim may not be nested inside an aggregate type}}
  %cast = trait.proj.cast %v, %c
    : tuple<!trait.proj<@Fold[i32], "Item">, i32>
    to tuple<!trait.claim<@U[i32]>, i32>
    by !trait.claim<@Fold[i32]>
  return
}
