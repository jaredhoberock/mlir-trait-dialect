// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds Item<U> to a pair of Item projections at two larger arguments, so
// each layer of the binding holds twice the projections of the one above it.
// A reading descends into what a binding spells one projection at a time,
// nested and bounded by the depth limit, so it meets the limit along the first
// branch and overflows there, named once, without reading the layers the
// branches double into.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>
}
// expected-error @below {{128 readings of projections' arguments and bindings stand nested}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }
