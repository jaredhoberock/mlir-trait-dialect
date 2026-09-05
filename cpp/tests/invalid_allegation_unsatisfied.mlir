// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation asserts a claim and names no evidence: monomorphization must
// find the impl that discharges it. Nothing implements @T for i32 here, so impl
// selection refuses the allegation where it stands and the round that follows
// reports it unresolved. An allegation that cannot be discharged is a compile
// error, never a claim admitted on its own say-so.

// CHECK: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@T[i32]>'
// CHECK: unresolved monomorphic trait.allege after resolve-impls

trait.trait private @T[!trait.poly<0>] {
}

func.func private @needs(%c: !trait.claim<@T[i32]>) {
  return
}

func.func @main() {
  %c = trait.allege @T[i32]
  func.call @needs(%c) : (!trait.claim<@T[i32]>) -> ()
  return
}
