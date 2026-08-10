// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A projection may not stand for a type that contains the projection itself: that
// closes a cycle -- an unfoundable infinite type no monomorphization can supply.
// An occurs check refuses it before the binding is made, so the verifier stays a
// total function rather than diverging on the resolution walk.

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

// The projection @Fold[i64]::Item asked to stand for tuple<@Fold[i64]::Item>.
func.func @self_wrap(%x: !trait.proj<@Fold[i64], "Item">)
    -> tuple<!trait.proj<@Fold[i64], "Item">> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x : !trait.proj<@Fold[i64], "Item">
    to tuple<!trait.proj<@Fold[i64], "Item">> unproven
  return %y : tuple<!trait.proj<@Fold[i64], "Item">>
}

// -----

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @I
  trait.assoc_type @J
}

// A tuple-position swap with one side wrapped: unifying binds I to tuple<J>, then
// J to tuple<J>, whose second binding closes the cycle on J.
func.func @swap_wrap(
    %x: tuple<!trait.proj<@Fold[i64], "I">, !trait.proj<@Fold[i64], "J">>)
    -> tuple<tuple<!trait.proj<@Fold[i64], "J">>, !trait.proj<@Fold[i64], "I">> {
  // expected-error @below {{are not consistent as a pending coerce}}
  %y = trait.coerce %x
    : tuple<!trait.proj<@Fold[i64], "I">, !trait.proj<@Fold[i64], "J">>
    to tuple<tuple<!trait.proj<@Fold[i64], "J">>, !trait.proj<@Fold[i64], "I">> unproven
  return %y : tuple<tuple<!trait.proj<@Fold[i64], "J">>, !trait.proj<@Fold[i64], "I">>
}
