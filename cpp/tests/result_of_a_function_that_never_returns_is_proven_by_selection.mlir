// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=ERASED

// A function that never returns declares an unproven claim as its result. No
// return hands back a value whose proof would settle it, so the result is a
// position no producer writes, as an external function's is, and selection
// writes it: the unconditional @I proves @B[i64], and the result leaves with
// erasure. Without @I it stands unproven
// (invalid_unproven_claim_in_the_result_of_a_function_that_never_returns.mlir).

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @I(%self: !trait.claim<@B[i64]>) {}

// CHECK-LABEL: func.func @f() -> !trait.claim<@B[i64] by @I>
// ERASED-LABEL: func.func @f() {
func.func @f() -> !trait.claim<@B[i64]> {
  cf.br ^loop
^loop:
  cf.br ^loop
}
