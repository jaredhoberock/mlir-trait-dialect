// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// isPendingExpansion counts a standing unproven monomorphic application claim
// outside a template as pending instantiation, so the erase gate it feeds does
// not admit erase before instantiate has discharged it. Here the method call's
// claim is unproven, so the call is not yet rewritable, yet the standing allege
// claim keeps instantiation pending.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s

trait.trait private @T[!trait.poly<0>] {
  func.func private @m(!trait.poly<0>) -> i32
}
trait.impl private @T_i32 for @T[i32] {
  func.func @m(%a: i32) -> i32 {
    %c = arith.constant 1 : i32
    return %c : i32
  }
}

// CHECK: method.call rewritable=false
// CHECK: pending-expansion=true
func.func @host(%x: i32) -> i32 {
  %ev = trait.allege @T[i32]
  %r = trait.method.call %ev @T[i32]::@m(%x) : (i32) -> i32
  return %r : i32
}
