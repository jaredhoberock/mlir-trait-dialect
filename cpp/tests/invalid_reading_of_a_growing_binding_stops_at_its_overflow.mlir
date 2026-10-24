// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Each binding below grows: through two associated types that spell each other,
// through two arguments that swap, and beneath a tuple inside its own argument.
// The reading overflows at a bound of its rounds -- the depth limit, or the
// width limit where the growth branches -- and is named once, and an overflow
// stops selection for the stage, so no reading goes on re-expanding what its
// walk had already answered.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
  trait.assoc_type @Step<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [!trait.proj<@A[i32], "Step", [!trait.poly<0>]>]>, !trait.proj<@A[i32], "Step", [!trait.proj<@A[i32], "Item", [!trait.poly<0>]>]>>
  trait.assoc_type @Step<[!trait.poly<0>]> = !trait.proj<@A[i32], "Item", [!trait.poly<0>]>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': 128 projection steps stand on the chain}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>, !trait.poly<2>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>, !trait.poly<1>]> = !trait.proj<@A[i32], "Item", [!trait.proj<@A[i32], "Item", [!trait.poly<0>, !trait.poly<1>]>, !trait.proj<@A[i32], "Item", [!trait.poly<1>, !trait.poly<0>]>]>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64, i32]> = i64>': 128 projection steps stand on the chain}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64, i32]> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.proj<@A[i32], "Item", [tuple<!trait.proj<@A[i32], "Item", [!trait.poly<0>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>]>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>': resolving its projections spells more than 2048 distinct types}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i64]> = i64>) { return }
