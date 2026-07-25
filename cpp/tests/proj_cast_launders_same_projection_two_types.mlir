// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The other vector the unproven-claim branch of the trait.proj.cast verifier
// closes: a single projection standing for two different concrete types across
// input and result. Both @Fold::Item projections map to one shared unification
// variable, so unifying the input against the result forces that variable to
// equal i32 (from the first tuple element) and i64 (from the second) at once.
// The conflict is rejected -- one resolution of @Fold::Item cannot be both.

!F = !trait.poly<0>

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

func.func @launders_same_projection_two_types(
    %v: tuple<!trait.proj<@Fold[!F], "Item">, i64>,
    %claim: !trait.claim<@Fold[!F]>)
    -> tuple<i32, !trait.proj<@Fold[!F], "Item">> {
  // expected-error @below {{are not consistent under claim}}
  %cast = trait.proj.cast %v, %claim
    : tuple<!trait.proj<@Fold[!F], "Item">, i64>
    to tuple<i32, !trait.proj<@Fold[!F], "Item">>
    by !trait.claim<@Fold[!F]>
  return %cast : tuple<i32, !trait.proj<@Fold[!F], "Item">>
}
