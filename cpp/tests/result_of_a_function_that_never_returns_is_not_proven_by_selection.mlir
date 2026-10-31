// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// A function that never returns declares an unproven claim as its result. No
// return hands back a value whose proof would settle it, and selection writes
// no position of a declaration: a claim a function states is proven by what
// its producer hands it, never selected for where it is declared. So the
// unconditional @I, which proves @B[i64], is never asked, and the result
// stands unproven, named where the function is declared, as it is with no
// impl at all
// (invalid_unproven_claim_in_the_result_of_a_function_that_never_returns.mlir).

trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>)
trait.impl private @I(%self: !trait.claim<@B[i64]>) {
}

// CHECK: :[[@LINE+1]]:1: error: unproven monomorphic claim '!trait.claim<@B[i64]>' after instantiate-monomorphs
func.func @f() -> !trait.claim<@B[i64]> {
  cf.br ^loop
^loop:
  cf.br ^loop
}
