// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s

// A marked (unproven) trait.coerce cites nothing and stands in the pending
// judgment: each !trait.proj term is a shared unification variable keyed by the
// projection itself, so a projection reconciles against the ground type an impl
// minted at monomorphization will supply. These forms verify, and the trailing
// `unproven` keyword prints so the marker survives a round trip (the printer
// emits no attribute dictionary).

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// A whole projection stands for the concrete type it will resolve to.
// CHECK-LABEL: func.func @proj_to_concrete
// CHECK: trait.coerce %{{.*}} : !trait.proj<@Fold[i64], "Item"> to i64 unproven
func.func @proj_to_concrete(%x: !trait.proj<@Fold[i64], "Item">) -> i64 {
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "Item"> to i64 unproven
  return %y : i64
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// A projection reconciles one tuple position while the rigid position matches
// literally: the pending mode absorbs the delta only where a projection sits.
// CHECK-LABEL: func.func @absorb_one_position
// CHECK: trait.coerce %{{.*}} unproven
func.func @absorb_one_position(%x: tuple<!trait.proj<@Fold[i64], "Item">, i64>)
    -> tuple<f64, i64> {
  %y = trait.coerce %x : tuple<!trait.proj<@Fold[i64], "Item">, i64>
    to tuple<f64, i64> unproven
  return %y : tuple<f64, i64>
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// The same projection on both sides is one variable, so the reflexive endpoints
// pass -- the discharged terminal the folder collapses once the projection
// respells to ground.
// CHECK-LABEL: func.func @same_projection
// CHECK: trait.coerce %{{.*}} unproven
func.func @same_projection(%x: !trait.proj<@Fold[i64], "Item">)
    -> !trait.proj<@Fold[i64], "Item"> {
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "Item">
    to !trait.proj<@Fold[i64], "Item"> unproven
  return %y : !trait.proj<@Fold[i64], "Item">
}
