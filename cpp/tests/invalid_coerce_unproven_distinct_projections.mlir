// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A projection is a shared unification variable keyed by the projection itself.
// It may resolve only to a projection-free position or stand for itself; a
// binding that still carries a distinct projection would equate two projections
// the pending form never licensed.

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

// Two distinct projections crossing: @Fold[i64]::A is asked to stand for
// @Fold[i64]::B, equating associated types the pending form does not justify.
func.func @distinct_projections(%x: !trait.proj<@Fold[i64], "A">)
    -> !trait.proj<@Fold[i64], "B"> {
  // expected-error @below {{equate distinct projections in a pending coerce}}
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "A">
    to !trait.proj<@Fold[i64], "B"> unproven
  return %y : !trait.proj<@Fold[i64], "B">
}
