// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -split-input-file %s | mlir-opt -split-input-file | FileCheck %s

// A marked coerce may relate two distinct projections: they alias into one
// equivalence class, two lookups asserted to denote one type, each still owed a
// projection-free grounding the minted impl supplies at discharge. The
// projections may stand bare on both sides, or one may stand for a composite
// that still carries projections -- every projection in that composite is
// itself owed a grounding, so the assertion is weaker than a ground terminal,
// not stronger. These forms verify and survive a round trip with the `unproven`
// marker intact.

trait.trait private @Fold[!trait.poly<0>] {
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

trait.trait private @Fold[!trait.poly<0>] {
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

trait.trait private @Conv[!trait.poly<0>, !trait.poly<1>] {}
trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// The aliased projections may sit nested inside an application claim's
// arguments -- decomposeTerm reads the attribute holding them directly, and the
// rigid position (i32) still matches literally. This is the shape a convergence
// respell presents.
// CHECK-LABEL: func.func @nested_in_application
// CHECK: trait.coerce %{{.*}} unproven
func.func @nested_in_application(
    %x: !trait.claim<@Conv[!trait.proj<@Fold[i64], "A">, i32]>)
    -> !trait.claim<@Conv[!trait.proj<@Fold[i64], "B">, i32]> {
  %y = trait.coerce %x : !trait.claim<@Conv[!trait.proj<@Fold[i64], "A">, i32]>
    to !trait.claim<@Conv[!trait.proj<@Fold[i64], "B">, i32]> unproven
  return %y : !trait.claim<@Conv[!trait.proj<@Fold[i64], "B">, i32]>
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// A bare projection may stand for a COMPOSITE that still carries a projection.
// This is one whole lookup standing for a spelling that carries a second
// lookup; both ground at the same monomorphization.
// CHECK-LABEL: func.func @projection_bearing_composite
// CHECK: trait.coerce %{{.*}} : !trait.proj<@Fold[i64], "A"> to tuple<!trait.proj<@Fold[i64], "B">> unproven
func.func @projection_bearing_composite(%x: !trait.proj<@Fold[i64], "A">)
    -> tuple<!trait.proj<@Fold[i64], "B">> {
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "A">
    to tuple<!trait.proj<@Fold[i64], "B">> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "B">>
}
