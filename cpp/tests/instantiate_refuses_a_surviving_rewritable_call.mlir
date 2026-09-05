// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// A generic call the rounds could rewrite but did not -- its result stays
// polymorphic because the arguments do not determine the callee's result
// variable, so specialization declines fail-closed -- and that stands outside
// every template is named at the instantiate exit rather than left for a later
// step to meet a call it cannot lower.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics

func.func private @foo(!trait.poly<0>) -> !trait.poly<1>

func.func @main(%x: i64) {
  // expected-error@+1 {{rewritable generic call survived instantiate-monomorphs}}
  %r = trait.func.call @foo(%x) : (i64) -> !trait.poly<9>
  return
}
