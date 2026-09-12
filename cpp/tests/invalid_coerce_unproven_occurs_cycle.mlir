// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A marked coerce whose endpoints still spell a projection stands: what each
// projection denotes is settled by the impls monomorphization mints, not by the
// verifier, and the erase barrier is where the two ground spellings are compared.
// These three are the shapes a spelling-level unifier used to refuse for closing
// a cycle -- a question about the spellings rather than about the program, since
// each projection resolves to whatever its impl binds and no infinite type is
// ever built.

trait.trait private @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// The projection @Fold[i64]::Item standing opposite tuple<@Fold[i64]::Item>.
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

// A tuple-position swap with one side wrapped.
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

// A chain of bare aliases closing into a composite on itself.
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
