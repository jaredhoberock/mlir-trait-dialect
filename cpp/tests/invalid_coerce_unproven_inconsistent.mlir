// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The pending mode is not an escape hatch. Two endpoints that are already
// ground are settled: no instantiation and no generated impl can bring them
// together, so a marked coerce over such a delta is refused exactly as an
// uncited one is.

// A ground structural collapse -- two positions folding to one, no projection
// anywhere -- is the shape a cross-group tensor reconciliation presents.
func.func @cross_group_collapse(%x: tuple<i64, i64>) -> i64 {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : tuple<i64, i64> to i64 unproven
  return %y : i64
}

// -----

// A ground claim opposite another ground claim: the two name different
// applications, and nothing later renames either.
trait.trait private @Red[!trait.poly<0>] {}
trait.trait private @Blue[!trait.poly<0>] {}

func.func @two_ground_claims(%x: !trait.claim<@Red[i64]>) -> !trait.claim<@Blue[i64]> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : !trait.claim<@Red[i64]> to !trait.claim<@Blue[i64]> unproven
  return %y : !trait.claim<@Blue[i64]>
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// An endpoint that still spells a projection is open, whatever the constructors
// above it do: what the projection denotes is settled by the impls
// monomorphization mints, and the erase barrier compares the two ground
// spellings.
func.func @open_delta_stands(%x: tuple<!trait.proj<@Fold[i64], "Item">, i64>)
    -> f64 {
  %y = trait.coerce %x : tuple<!trait.proj<@Fold[i64], "Item">, i64> to f64 unproven
  return %y : f64
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

// The same, with the rigid delta beside the projection rather than above it.
func.func @open_delta_beside_a_rigid_position(
    %x: tuple<!trait.proj<@Fold[i64], "A">, tuple<i32>>)
    -> tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>> {
  %y = trait.coerce %x : tuple<!trait.proj<@Fold[i64], "A">, tuple<i32>>
    to tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>>
}
