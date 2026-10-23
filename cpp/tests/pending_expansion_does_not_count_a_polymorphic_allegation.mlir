// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s

// An allegation over a type variable outside a template is no work
// instantiation can do: it proves monomorphic allegations only, and outside a
// template no instance substitutes the variable. The module is not pending, so
// the lowering driver hands it to erasure, whose entry refuses the claim by
// name (invalid_claim_carrying_a_type_variable_outside_a_template_is_refused_at_erasure.mlir),
// rather than running instantiate on work it leaves standing.

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_i64(%self: !trait.claim<@B[i64]>) {}

// CHECK: pending-expansion=false
func.func @alleged() -> i64 {
  %a = trait.allege @B[!trait.poly<0>]
  %c = arith.constant 7 : i64
  return %c : i64
}
