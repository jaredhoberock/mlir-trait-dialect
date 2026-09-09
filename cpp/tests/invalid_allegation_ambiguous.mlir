// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Two unconditional impls satisfy @T[i32]. An allegation names no preferred
// impl, so it must report incoherence and remain unresolved at the phase boundary.

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

// CHECK: 'trait.allege' op incoherent impls (multiple satisfiable) for '!trait.claim<@T[i32]>'
// CHECK: unresolved monomorphic trait.allege after resolve-impls
