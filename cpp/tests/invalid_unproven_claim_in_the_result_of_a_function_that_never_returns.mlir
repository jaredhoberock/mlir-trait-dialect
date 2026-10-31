// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s --check-prefix=READY
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s --check-prefix=ERASE
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --check-prefix=MONO

// A function that never returns declares an unproven claim as its result. No
// return hands back a value whose proof could settle it, so the claim stands
// in the signature alone, and it is work instantiation owes there: the module
// is pending instantiation, nothing selects for a declaration's position, so
// instantiation names the claim unproven, and erasure handed the module alone
// refuses it as owed. An impl serving the claim changes none of that
// (result_of_a_function_that_never_returns_is_not_proven_by_selection.mlir).

trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>)

// READY: pending-expansion=true
// ERASE: error: unproven monomorphic claim '!trait.claim<@B[i64]>' before erasure
// MONO-NOT: no impl with satisfiable assumptions
// MONO: error: unproven monomorphic claim '!trait.claim<@B[i64]>' after instantiate-monomorphs
func.func @f() -> !trait.claim<@B[i64]> {
  cf.br ^loop
^loop:
  cf.br ^loop
}
