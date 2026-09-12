// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A generic callee declared without a body has nothing to clone, so the call
// cannot be instantiated. Specialization refuses it at the declaration, the
// lowering pattern hands the refusal back rather than reading an instance that
// was never built, and the call is left standing for the exit check to name.

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// CHECK: error: cannot specialize external function
// CHECK: error: 'trait.func.call' op rewritable generic call survived instantiate-monomorphs
func.func private @external(%x: !trait.poly<0>) -> !trait.poly<0>

func.func @main(%y: i64) -> i64 {
  %z = trait.func.call @external(%y) {type_params = [!trait.poly<0>], type_args = [i64]} : (i64) -> i64
  return %z : i64
}
