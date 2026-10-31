// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A generic call names a function of the symbol table it stands in, as a
// func.call does, so the private gpu.module here contains the template and the
// callee its call names. A generic call inside a template is never lowered
// where it stands: a template no instance reaches is collected with its calls,
// and the gpu.module, containing nothing else, is collected with it.

// VERIFY: gpu.module @g
// VERIFY: trait.func.call @f
gpu.module @g attributes {sym_visibility = "private"} {
  func.func private @f(%x: !trait.poly<0>) -> !trait.poly<0> { return %x : !trait.poly<0> }
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
