// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s

// A marked coerce may relate two distinct BARE projections: they alias into one
// equivalence class, two lookups asserted to denote one type, each still owed a
// projection-free grounding the minted impl supplies at discharge. This is the
// direct-alias form the pending judgment admits alongside a projection standing
// for a concrete type; a projection-bearing composite is still refused (see
// invalid_coerce_unproven_distinct_projections). These forms verify and survive
// a round trip with the `unproven` marker intact.

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// Two distinct projections aliased directly.
// CHECK-LABEL: func.func @bare_alias
// CHECK: trait.coerce %{{.*}} : !trait.proj<@Fold[i64], "A"> to !trait.proj<@Fold[i64], "B"> unproven
func.func @bare_alias(%x: !trait.proj<@Fold[i64], "A">)
    -> !trait.proj<@Fold[i64], "B"> {
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "A">
    to !trait.proj<@Fold[i64], "B"> unproven
  return %y : !trait.proj<@Fold[i64], "B">
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// The alias is symmetric: the reversed orientation is the same equivalence.
// CHECK-LABEL: func.func @bare_alias_reversed
// CHECK: trait.coerce %{{.*}} : !trait.proj<@Fold[i64], "B"> to !trait.proj<@Fold[i64], "A"> unproven
func.func @bare_alias_reversed(%x: !trait.proj<@Fold[i64], "B">)
    -> !trait.proj<@Fold[i64], "A"> {
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "B">
    to !trait.proj<@Fold[i64], "A"> unproven
  return %y : !trait.proj<@Fold[i64], "A">
}

// -----

trait.trait @Conv[!trait.poly<0>, !trait.poly<1>] {}
trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// The aliased projections may sit nested inside an application claim's
// arguments -- decomposeTerm reaches the hand-written attribute storage the
// generic walkers are opaque to, and the rigid position (i32) still matches
// literally. This is the shape a convergence respell presents.
// CHECK-LABEL: func.func @nested_in_application
// CHECK: trait.coerce %{{.*}} unproven
func.func @nested_in_application(
    %x: !trait.claim<@Conv[!trait.proj<@Fold[i64], "A">, i32]>)
    -> !trait.claim<@Conv[!trait.proj<@Fold[i64], "B">, i32]> {
  %y = trait.coerce %x : !trait.claim<@Conv[!trait.proj<@Fold[i64], "A">, i32]>
    to !trait.claim<@Conv[!trait.proj<@Fold[i64], "B">, i32]> unproven
  return %y : !trait.claim<@Conv[!trait.proj<@Fold[i64], "B">, i32]>
}
