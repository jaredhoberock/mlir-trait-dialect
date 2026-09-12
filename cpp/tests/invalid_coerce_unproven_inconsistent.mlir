// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The pending mode is not an escape hatch. Two endpoints that are already
// ground are settled: no instantiation and no generated impl can bring them
// together, so a marked coerce over such a delta is refused exactly as an
// uncited one is.
//
// The last two sections carry the impl that settles their projections, so the
// second run says what becomes of a delta the verifier let stand: the spellings
// settle, the delta is still there, and the coerce is refused. Neither
// reconciles -- a rigid position the projection does not stand in already
// disagrees.

// A ground structural collapse -- two positions folding to one, no projection
// anywhere -- is the shape a cross-group tensor reconciliation presents.
// CHECK: input type 'tuple<i64, i64>' and result type 'i64' are not consistent as a pending coerce
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

// CHECK: input type '!trait.claim<@Red[i64]>' and result type '!trait.claim<@Blue[i64]>' are not consistent as a pending coerce
func.func @two_ground_claims(%x: !trait.claim<@Red[i64]>) -> !trait.claim<@Blue[i64]> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : !trait.claim<@Red[i64]> to !trait.claim<@Blue[i64]> unproven
  return %y : !trait.claim<@Blue[i64]>
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @Item = i32
}

// An endpoint that still spells a projection is open, whatever the constructors
// above it do: what the projection denotes is settled by the impls
// monomorphization mints, and the two ground spellings are compared then.
// CHECK: input type 'tuple<i32, i64>' and result type 'f64' are not consistent as a pending coerce
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

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @A = i8
  trait.assoc_type @B = i8
}

// The same, with the rigid delta beside the projection rather than above it.
// CHECK: input type 'tuple<i8, tuple<i32>>' and result type 'tuple<i8, tuple<i64>>' are not consistent as a pending coerce
func.func @open_delta_beside_a_rigid_position(
    %x: tuple<!trait.proj<@Fold[i64], "A">, tuple<i32>>)
    -> tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>> {
  %y = trait.coerce %x : tuple<!trait.proj<@Fold[i64], "A">, tuple<i32>>
    to tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "B">, tuple<i64>>
}
