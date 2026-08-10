// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A projection is a shared unification variable keyed by the projection itself.
// It may resolve to a projection-free position, stand for itself, or alias
// another bare projection; what it may NOT resolve to is a composite still
// carrying a projection, nor stand for two concrete types at once.

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// One projection standing for two concrete types: @Fold[i64]::Item is one
// variable, forced to equal i32 in the first position and i64 in the second.
func.func @one_projection_two_types(
    %x: tuple<!trait.proj<@Fold[i64], "Item">, !trait.proj<@Fold[i64], "Item">>)
    -> tuple<i32, i64> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "Item">, !trait.proj<@Fold[i64], "Item">>
    to tuple<i32, i64> unproven
  return %y : tuple<i32, i64>
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// A bare alias @Fold[i64]::A <-> @Fold[i64]::B is licensed (both owed a
// grounding at discharge), but a projection may not resolve to a COMPOSITE that
// still carries a projection: @Fold[i64]::A asked to stand for
// tuple<@Fold[i64]::B> would equate two distinct projections inside a rigid
// constructor, a shape the pending form never licensed.
func.func @projection_bearing_composite(%x: !trait.proj<@Fold[i64], "A">)
    -> tuple<!trait.proj<@Fold[i64], "B">> {
  // expected-error @below {{equate distinct projections in a pending coerce}}
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "A">
    to tuple<!trait.proj<@Fold[i64], "B">> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "B">>
}
