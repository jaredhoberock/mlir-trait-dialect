// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds Item to a value of two tuples around Item itself, and a value's
// type spells Item. Each round of the reading wraps two layers more, so it
// meets the rounds' depth limit and overflows there, named once: the reading
// never settles on a form still spelling Item, which the stage would then take
// for the value's type and rewrite again.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.impl private @I(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item = tuple<tuple<!trait.proj<@A[i32], "Item">>> }
// expected-error @below {{overflow evaluating the requirement '!trait.proj<@A[i32], "Item">': 128 projection steps stand on the chain}}
func.func private @use(!trait.proj<@A[i32], "Item">)

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item<[!trait.poly<1>]> }
trait.impl private @I(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item<[!trait.poly<0>]> = tuple<tuple<!trait.proj<@A[i32], "Item", [!trait.poly<0>]>>> }
// expected-error @below {{overflow evaluating the requirement '!trait.proj<@A[i32], "Item", [i64]>': 128 projection steps stand on the chain}}
func.func private @use(!trait.proj<@A[i32], "Item", [i64]>)
