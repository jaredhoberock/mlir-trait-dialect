// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds Item<U> to a pair of Item projections at two larger arguments, so
// each layer of the binding holds twice the projections of the one above it,
// whichever branch comes first. A reading reads a layer a round, and the
// distinct types it spells double every round: it meets the width limit long
// before the rounds' depth limit and overflows there, named once by what it
// spells as written.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': resolving its projections spells more than 2048 distinct types}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': resolving its projections spells more than 2048 distinct types}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }

// -----

// One branch, doubling its argument: the reading grows a layer a round and
// overflows at the rounds' depth limit. Its partial normal form spells a tree
// of 2^128 leaves in a few hundred distinct types, so it is named by what it
// spells as written, never by the partial form.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': 128 projection steps stand on the chain}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }
