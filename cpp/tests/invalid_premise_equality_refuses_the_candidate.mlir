// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' 2>&1 | FileCheck %s

// An impl's where-clause equality says when the impl applies, so selection is
// what discharges it. @X_gen applies only where @Ten[T]::Shape is i64; at i32
// that projection resolves to tuple<i64, i64>, so the premise does not hold,
// @X_gen is no candidate, and the demand @X[i32] has no impl.

// CHECK: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@X[i32]>'

!T = !trait.poly<0>

trait.trait private @Ten[!T] {
  trait.assoc_type @Shape
}

trait.impl private @Ten_i32 for @Ten[i32] {
  trait.assoc_type @Shape = tuple<i64, i64>
}

trait.trait private @X[!T] {}

!U = !trait.poly<1>
trait.impl private @X_gen for @X[!U] where [!trait.proj<@Ten[!U], "Shape"> = i64] {}

func.func private @needs(!trait.claim<@X[i32]>)

func.func @main() {
  %c = trait.allege @X[i32]
  trait.func.call @needs(%c) : (!trait.claim<@X[i32]>) -> ()
  return
}
