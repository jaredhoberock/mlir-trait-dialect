// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// One projection standing opposite two different types is a claim about what
// the impls monomorphization mints will bind, and the verifier holds none of
// them. So a marked coerce whose endpoints still spell a projection stands here
// and is judged at the erase barrier, where both spellings are ground and an
// undischarged coerce is refused.

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// @Fold[i64]::Item opposite i32 in one position and i64 in the other.
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

// The same shape reached through a projection-bearing composite.
func.func @composite_binding_then_rigid_mismatch(
    %x: tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "A">>)
    -> tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64> {
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "A">>
    to tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64> unproven
  return %y : tuple<tuple<!trait.proj<@Fold[i64], "B">>, i64>
}
