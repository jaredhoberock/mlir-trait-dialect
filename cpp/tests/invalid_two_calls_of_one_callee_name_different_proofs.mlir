// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// One callee, one type argument, two calls whose claim operands name different
// proofs of @T[i64]. The callee is specialized once for i64, and its parameter
// carries whichever proof the instance was cut with, so the other call passes a
// claim the clone's signature does not take. The refusal stands at the call
// that was rewritten.

// CHECK: error: 'func.call' op operand type mismatch: expected operand type '!trait.claim<@T[i64] by @pv2>', but provided '!trait.claim<@T[i64] by @pv1>' for operand number 0
// CHECK-NEXT: trait.func.call @g(%w1)

trait.trait private @T[!trait.poly<0>] { func.func private @t() -> i64 }
trait.impl private @T_i64 for @T[i64] {
  func.func @t() -> i64 {
    %c = arith.constant 5 : i64
    return %c : i64
  }
}
trait.proof private @pv1 proves @T_i64 for @T[i64] given []
trait.proof private @pv2 proves @T_i64 for @T[i64] given []

func.func private @g(%c: !trait.claim<@T[!trait.poly<3>]>) -> i64 {
  %r = trait.method.call %c @T[!trait.poly<3>]::@t() : () -> i64
  return %r : i64
}

func.func @main() -> i64 {
  %w1 = trait.witness @pv1 for @T[i64]
  %w2 = trait.witness @pv2 for @T[i64]
  %a = trait.func.call @g(%w1) {type_params = [!trait.poly<3>], type_args = [i64]}
    : (!trait.claim<@T[i64] by @pv1>) -> i64
  %b = trait.func.call @g(%w2) {type_params = [!trait.poly<3>], type_args = [i64]}
    : (!trait.claim<@T[i64] by @pv2>) -> i64
  %s = arith.addi %a, %b : i64
  return %s : i64
}
