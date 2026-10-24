// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds @A[T]::Out to a projection of Out over @A at @A[T]::Out itself:
// the binding grows its own head application's argument. A projection is read
// once its head's arguments are, as once its own are, a layer a round, so the
// growth meets the rounds' depth limit and overflows there, named once, rather
// than nesting selections of the head without end.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out }
trait.impl private @I(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out = !trait.proj<@A[!trait.proj<@A[!T], "Out">], "Out"> }
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i64], "Out"> = i64>': 128 projection steps stand on the chain}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i64], "Out"> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out }
trait.impl private @I(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out = !trait.proj<@A[tuple<!trait.proj<@A[!T], "Out">, !trait.proj<@A[!T], "Out">>], "Out"> }
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i64], "Out"> = i64>': 128 projection steps stand on the chain}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i64], "Out"> = i64>) { return }
