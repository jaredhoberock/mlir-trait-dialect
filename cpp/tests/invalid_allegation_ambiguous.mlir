// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Two unconditional impls satisfy @T[i32]. An allegation names no preferred
// impl, so it must report incoherence and stand unproven at the stage's exit.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {
}

trait.impl private @T_first(%self: !trait.claim<@T[i32]>) {
}

trait.impl private @T_second(%self: !trait.claim<@T[i32]>) {
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
// CHECK: unproven monomorphic claim '!trait.claim<@T[i32]>' after instantiate-monomorphs
