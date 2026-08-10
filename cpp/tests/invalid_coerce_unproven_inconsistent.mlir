// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The pending mode is not an escape hatch. With every non-projection position
// rigid, endpoints that differ where no projection sits do not unify, so a
// marked coerce over such a delta is refused exactly as an uncited one is.

// A ground structural collapse -- two positions folding to one, no projection
// anywhere -- is the shape a cross-group tensor reconciliation presents. It has
// no projection position to stand for the difference, so it must refuse.
func.func @cross_group_collapse(%x: tuple<i64, i64>) -> i64 {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : tuple<i64, i64> to i64 unproven
  return %y : i64
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// An unabsorbable delta: the input is a tuple carrying a projection, the result
// a scalar. The constructors diverge above any projection, so no projection
// position can stand for the difference.
func.func @unabsorbable_delta(%x: tuple<!trait.proj<@Fold[i64], "Item">, i64>)
    -> f64 {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : tuple<!trait.proj<@Fold[i64], "Item">, i64> to f64 unproven
  return %y : f64
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// A licensed bare alias in one position does not relax the rigid position beside
// it: @Fold[i64]::A aliases @Fold[i64]::B in the first tuple slot, but the second
// slot pairs rigid tuple<i32> against tuple<i64>. Aliasing projections never
// equates tokens standing where no projection sits, so the delta is refused.
func.func @alias_does_not_relax_rigid(
    %x: tuple<!trait.proj<@Fold[i64], "A">, tuple<i32>>)
    -> tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : tuple<!trait.proj<@Fold[i64], "A">, tuple<i32>>
    to tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>>
}
