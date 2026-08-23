// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A projection is a shared unification variable keyed by the projection itself,
// so it stands for at most one type. Forcing one projection to equal two
// different rigid types at once is the inconsistency the pending judgment
// refuses.

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

// The same inconsistency reached through a projection-bearing composite: @A is
// bound to tuple<@B> in the first position, so the second position forces @B to
// be both i32 and, transitively through @A's binding, the rigid i64 -- the
// tuple constructor cannot match i64.
func.func @composite_binding_then_rigid_mismatch(
    %x: tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "A">>)
    -> tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "A">>
    to tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64> unproven
  return %y : tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64>
}
