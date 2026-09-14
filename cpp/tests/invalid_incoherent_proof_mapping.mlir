// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// One semantic application admits one proof symbol. Recording two claims for
// the same obligation is coherent exactly when they name the same proof; a
// second, different symbol for it is refused where the call's substitution is
// built, because a call carries one spelling of an application into its callee
// and two proofs of @T[i64] give it two. Two impls exist for @T[i64], so the
// proof symbol -- not the application -- is what makes a recording coherent.

// CHECK: error: 'trait.func.call' op inconsistent proof mapping: '!trait.claim<@T[i64]>' is already bound to '!trait.claim<@T[i64] by @T_a>', but attempted to bind '!trait.claim<@T[i64] by @T_b>'
// CHECK-NEXT: trait.func.call @callee(%a, %b)

trait.trait private @T[!trait.poly<0>] {}

trait.impl private @T_a for @T[i64] {}
trait.impl private @T_b for @T[i64] {}

func.func private @callee(!trait.claim<@T[i64]>, !trait.claim<@T[i64]>)

// Coherent: both claims name the same proof symbol @T_a, so the second
// recording matches the first and is accepted.
func.func @coherent() {
  %a = trait.witness @T_a for @T[i64]
  trait.func.call @callee(%a, %a)
    : (!trait.claim<@T[i64] by @T_a>, !trait.claim<@T[i64] by @T_a>) -> ()
  return
}

// Incoherent: the two claims name different proof symbols for the one
// application, so the second recording conflicts and is refused.
func.func @incoherent() {
  %a = trait.witness @T_a for @T[i64]
  %b = trait.witness @T_b for @T[i64]
  trait.func.call @callee(%a, %b)
    : (!trait.claim<@T[i64] by @T_a>, !trait.claim<@T[i64] by @T_b>) -> ()
  return
}
