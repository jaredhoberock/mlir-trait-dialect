// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @callee binds no type parameter, so it is no template and has no instance to
// cut: a call reaches it as written. It takes @T[i64] through @T_a, and a call
// supplying @T[i64] through @T_b supplies evidence it does not take, which is
// refused where the call is lowered.

// CHECK: error: 'trait.func.call' op passes '!trait.claim<@T[i64] by @T_b>' as operand #0 to '@callee', which takes '!trait.claim<@T[i64] by @T_a>'
// CHECK-NEXT: trait.func.call @callee(%b)

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @T_a(%self: !trait.claim<@T[i64]>) {}
trait.impl private @T_b(%self: !trait.claim<@T[i64]>) {}

func.func private @callee(!trait.claim<@T[i64] by @T_a>)

func.func @f() {
  %b = trait.witness @T_b for @T[i64]
  trait.func.call @callee(%b) : (!trait.claim<@T[i64] by @T_b>) -> ()
  return
}
