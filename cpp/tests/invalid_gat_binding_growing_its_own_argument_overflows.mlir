// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds Item<U> to Item<Item<U>>, so each reading of a projection of Item
// reads its argument, itself a projection of Item, whose binding nests one
// more. What a step answers is read one step deeper, so the chain reaches the
// depth limit: the hypothesis overflows there, named once by what it spells as
// written, rather than nesting without end.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.proj<@A[i32], "Item", [!trait.proj<@A[i32], "Item", [!trait.poly<0>]>]>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }
