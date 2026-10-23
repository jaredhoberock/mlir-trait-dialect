// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A generic call inside a template is never lowered where it stands, so the
// rule refusing a call inside a symbol table other than its module
// (invalid_trait_call_inside_another_symbol_table.mlir) does not reach it: a
// template no instance reaches is collected with its calls, here the private
// gpu.module holding it with it.

// VERIFY: gpu.module @g
// VERIFY: trait.func.call @f
func.func private @f(%x: !trait.poly<0>) -> !trait.poly<0> { return %x : !trait.poly<0> }
gpu.module @g attributes {sym_visibility = "private"} {
  func.func private @unused(%x: !trait.poly<0>) -> !trait.poly<0> {
    %r = trait.func.call @f(%x) : (!trait.poly<0>) -> !trait.poly<0>
    return %r : !trait.poly<0>
  }
}

// CHECK-NOT: gpu.module
// CHECK-LABEL: func.func @run() -> i64
func.func @run() -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}
