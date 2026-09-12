// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A marked coerce whose endpoints still spell a projection stands: what each
// projection denotes is settled by the impls monomorphization mints, not by the
// verifier, and the erase barrier is where the two ground spellings are compared.
// These three are the shapes a spelling-level unifier used to refuse for closing
// a cycle -- a question about the spellings rather than about the program, since
// each projection resolves to whatever its impl binds and no infinite type is
// ever built.
//
// Each section carries the impl that settles its projections, so the second run
// is the other half of the verdict: once the spellings are settled all three
// deltas are real, and each coerce is refused. None of the three reconciles --
// a type cannot be the tuple it stands inside.

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @Item = i64
}

// The projection @Fold[i64]::Item standing opposite tuple<@Fold[i64]::Item>.
// CHECK: input type 'i64' and result type 'tuple<i64>' are not consistent as a pending coerce
func.func @self_wrap(%x: !trait.proj<@Fold[i64], "Item">)
    -> tuple<!trait.proj<@Fold[i64], "Item">> {
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "Item">
    to tuple<!trait.proj<@Fold[i64], "Item">> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "Item">>
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @I
  trait.assoc_type @J
}

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @I = i32
  trait.assoc_type @J = i64
}

// A tuple-position swap with one side wrapped.
// CHECK: input type 'tuple<i32, i64>' and result type 'tuple<tuple<i64>, i32>' are not consistent as a pending coerce
func.func @swap_wrap(
    %x: tuple<!trait.proj<@Fold[i64], "I">, !trait.proj<@Fold[i64], "J">>)
    -> tuple<tuple<!trait.proj<@Fold[i64], "J">>, !trait.proj<@Fold[i64], "I">> {
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "I">, !trait.proj<@Fold[i64], "J">>
    to tuple<tuple<!trait.proj<@Fold[i64], "J">>, !trait.proj<@Fold[i64], "I">> unproven
  return %y : tuple<tuple<!trait.proj<@Fold[i64], "J">>, !trait.proj<@Fold[i64], "I">>
}

// -----

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
  trait.assoc_type @C
}

trait.impl private @Fold_i64 for @Fold[i64] {
  trait.assoc_type @A = i8
  trait.assoc_type @B = i16
  trait.assoc_type @C = i32
}

// A chain of bare aliases closing into a composite on itself.
// CHECK: input type 'tuple<i8, i16, i32>' and result type 'tuple<i16, i32, tuple<i8>>' are not consistent as a pending coerce
func.func @alias_chain_into_composite(
    %x: tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "B">,
              !trait.proj<@Fold[i64], "C">>)
    -> tuple<!trait.proj<@Fold[i64], "B">, !trait.proj<@Fold[i64], "C">,
             tuple<!trait.proj<@Fold[i64], "A">>> {
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "A">, !trait.proj<@Fold[i64], "B">,
            !trait.proj<@Fold[i64], "C">>
    to tuple<!trait.proj<@Fold[i64], "B">, !trait.proj<@Fold[i64], "C">,
             tuple<!trait.proj<@Fold[i64], "A">>> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "B">, !trait.proj<@Fold[i64], "C">,
                    tuple<!trait.proj<@Fold[i64], "A">>>
}
