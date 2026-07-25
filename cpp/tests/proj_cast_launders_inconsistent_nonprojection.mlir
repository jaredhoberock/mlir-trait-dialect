// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// The unproven-claim branch of the trait.proj.cast verifier replaces each
// projection over the claim's trait application with a shared unification
// variable and unifies input against result, so the two must agree everywhere
// else. Here the claim trait @Fold does appear in a projection, but input and
// result differ in a non-projection tuple element (i32 vs f32) that no
// resolution of @Fold::Item could reconcile. Setting the @Fold projections aside
// leaves tuple<_, i32> against tuple<_, f32>, which does not unify, so the cast
// is rejected.

!F = !trait.poly<0>

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

func.func @launders_nonprojection(
    %v: tuple<!trait.proj<@Fold[!F], "Item">, i32>,
    %claim: !trait.claim<@Fold[!F]>)
    -> tuple<!trait.proj<@Fold[!F], "Item">, f32> {
  // expected-error @below {{are not consistent under claim}}
  %cast = trait.proj.cast %v, %claim
    : tuple<!trait.proj<@Fold[!F], "Item">, i32>
    to tuple<!trait.proj<@Fold[!F], "Item">, f32>
    by !trait.claim<@Fold[!F]>
  return %cast : tuple<!trait.proj<@Fold[!F], "Item">, f32>
}
