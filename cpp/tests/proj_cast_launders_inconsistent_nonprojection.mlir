// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Records the reach of the unproven-claim branch of the trait.proj.cast
// verifier. That branch checks only that the claim's trait appears in some
// projection of the input or result type; it never checks that input and result
// agree once those projections are set aside. Here the claim trait @Fold does
// appear in a projection, so the cast is accepted even though input and result
// differ in a non-projection tuple element (i32 vs f32) that no resolution of
// @Fold::Item could reconcile. Replacing each @Fold projection with a fresh
// unification variable and unifying input against result would surface the
// i32-vs-f32 conflict and reject the cast. Pinned as accepted so a stronger
// check registers as a change here.

!F = !trait.poly<0>

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// CHECK-LABEL: func.func @launders_nonprojection
// CHECK: trait.proj.cast
func.func @launders_nonprojection(
    %v: tuple<!trait.proj<@Fold[!F], "Item">, i32>,
    %claim: !trait.claim<@Fold[!F]>)
    -> tuple<!trait.proj<@Fold[!F], "Item">, f32> {
  %cast = trait.proj.cast %v, %claim
    : tuple<!trait.proj<@Fold[!F], "Item">, i32>
    to tuple<!trait.proj<@Fold[!F], "Item">, f32>
    by !trait.claim<@Fold[!F]>
  return %cast : tuple<!trait.proj<@Fold[!F], "Item">, f32>
}
