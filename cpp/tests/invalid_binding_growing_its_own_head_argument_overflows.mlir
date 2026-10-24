// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds @A[T]::Out to a projection of Out over @A at @A[T]::Out itself:
// the binding grows its own head application's argument. A projection's head
// arguments are read before it is, as its own arguments are, and what a step
// answers is read one step deeper, so the growth reaches the depth limit and
// overflows there, named once, rather than nesting selections of the head
// without end.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out }
trait.impl private @I(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out = !trait.proj<@A[!trait.proj<@A[!T], "Out">], "Out"> }
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i64], "Out"> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i64], "Out"> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out }
trait.impl private @I(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Out = !trait.proj<@A[tuple<!trait.proj<@A[!T], "Out">, !trait.proj<@A[!T], "Out">>], "Out"> }
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i64], "Out"> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i64], "Out"> = i64>) { return }
