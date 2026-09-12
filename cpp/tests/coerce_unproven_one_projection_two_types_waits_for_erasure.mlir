// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// One projection standing opposite two different types is a claim about what
// the impls monomorphization mints will bind, and the verifier holds none of
// them. So a marked coerce whose endpoints still spell a projection stands here
// and is judged once both spellings are settled, where an undischarged coerce is
// refused.
//
// Each section carries the impl that settles its projections, so the second run
// is the other half of the verdict. Neither reconciles: one projection denotes
// one type, so it cannot meet two different rigid positions at once.

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @Item = i32
}

// @Fold[i64]::Item opposite i32 in one position and i64 in the other.
// CHECK: input type 'tuple<i32, i32>' and result type 'tuple<i32, i64>' are not consistent as a pending coerce
func.func @one_projection_two_types(
    %x: tuple<!trait.proj<@Fold[i64], "Item">, !trait.proj<@Fold[i64], "Item">>)
    -> tuple<i32, i64> {
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "Item">, !trait.proj<@Fold[i64], "Item">>
    to tuple<i32, i64> unproven
  return %y : tuple<i32, i64>
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @A = i64
  trait.assoc_type @B = i64
}

// The same shape reached through a projection-bearing composite.
// CHECK: input type 'tuple<i64, i64>' and result type 'tuple<tuple<i64>, i64>' are not consistent as a pending coerce
func.func @composite_binding_then_rigid_mismatch(
    %x: tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "A">>)
    -> tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64> {
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "A">>
    to tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64> unproven
  return %y : tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64>
}
