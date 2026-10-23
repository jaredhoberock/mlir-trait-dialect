// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @f states the equality of its parameter with itself and uses it nowhere. A
// call supplying evidence as the parameter cuts an instance whose equality
// relates claims: the clone builds it, as it builds any substitution, and
// nothing reads it, so it leaves with the dead code instantiation collects and
// the program lowers. invalid_equality_a_substitution_states_over_evidence_is_refused.mlir
// is the twin whose instance passes the equality on.

trait.trait private @T(%s: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @I(%s: !trait.claim<@T[i32]>) {}
func.func private @f(%x: !trait.poly<0>) -> !trait.poly<0> {
  %e = trait.witness refl : !trait.claim<!trait.poly<0> = !trait.poly<0>>
  return %x : !trait.poly<0>
}

// CHECK-LABEL: func.func @run() -> i64
// CHECK: call @f_
// CHECK-NOT: trait.
func.func @run() -> i64 {
  %w = trait.witness @I for @T[i32]
  %c = trait.coerce %w : !trait.claim<@T[i32] by @I> to !trait.claim<@T[i32]>
  %r = trait.func.call @f(%c) : (!trait.claim<@T[i32]>) -> !trait.claim<@T[i32]>
  %k = arith.constant 7 : i64
  return %k : i64
}
