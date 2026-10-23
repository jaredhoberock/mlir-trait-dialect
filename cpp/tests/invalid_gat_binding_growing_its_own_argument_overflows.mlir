// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds Item<U> to Item<Item<U>>, so each reading of a projection of Item
// reads its argument, itself a projection of Item, whose binding nests one
// more. A projection's arguments are read by a reading nested in the
// projection's, and the nesting is bounded by the depth limit: the hypothesis
// overflows at the bound, named once with the readings it counts, rather than
// nesting without end.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.proj<@A[i32], "Item", [!trait.proj<@A[i32], "Item", [!trait.poly<0>]>]>
}
// expected-error @below {{overflow evaluating the requirement '!trait.proj<@A[i32], "Item", [i64]>': 128 readings of projections' arguments and bindings stand nested}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }
