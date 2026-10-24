// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I binds @A[T]::Item to a one-tuple of @A[tuple<T>]::Item: no generic
// associated type, and the binding grows beneath a constructor. Whatever reads
// @A[i64]::Item -- a hypothesis erasure judges, an allegation proved, a value's
// type -- reads what the binding spells one step deeper, a step a layer, to
// the depth limit, and overflows there, named once by what it spells as
// written.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.impl private @I(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item = tuple<!trait.proj<@A[tuple<!T>], "Item">>
}
// expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i64], "Item"> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.impl private @I(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item = tuple<!trait.proj<@A[tuple<!T>], "Item">>
}
func.func private @need(!trait.claim<!trait.proj<@A[i64], "Item"> = i64>)
func.func @run() {
  // expected-error @below {{overflow evaluating the requirement '!trait.claim<!trait.proj<@A[i64], "Item"> = i64>': a chain of steps resolving its projections does not end within 128 steps}}
  %e = trait.allege !trait.proj<@A[i64], "Item"> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> ()
  return
}

// -----

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
trait.impl private @I(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item = tuple<!trait.proj<@A[tuple<!T>], "Item">>
}
// expected-error @below {{overflow evaluating the requirement '!trait.proj<@A[i64], "Item">': a chain of steps resolving its projections does not end within 128 steps}}
func.func private @use(!trait.proj<@A[i64], "Item">)
