// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// An allegation asks impl selection which impl discharges its claim, and two
// unconditional impls binding @T[i32] leave that question with two answers. The
// refusal is selection's, not the allegation's: an allegation names no impl, so
// it can neither prefer one nor stand while both remain. The census records the
// refusal under its own arm, so what the run refused is legible beside the
// diagnostic.

// CHECK: 'trait.allege' op incoherent impls (multiple satisfiable) for '!trait.claim<@T[i32]>'
// CHECK: unresolved monomorphic trait.allege after resolve-impls
// CHECK: trait-stage-record digest {{.*}} refusals-ambiguous=1

trait.trait private @T[!trait.poly<0>] {
}

trait.impl private @T_first for @T[i32] {
}

trait.impl private @T_second for @T[i32] {
}

func.func private @needs(%c: !trait.claim<@T[i32]>) {
  return
}

func.func @main() {
  %c = trait.allege @T[i32]
  func.call @needs(%c) : (!trait.claim<@T[i32]>) -> ()
  return
}
