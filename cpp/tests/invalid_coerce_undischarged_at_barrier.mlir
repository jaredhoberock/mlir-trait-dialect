// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false %s 2>&1 | FileCheck %s

// A used coerce whose endpoints still differ after the barrier's conversion is
// not discharged: forwarding its input would change the value's written type
// with no evidence left to justify it. The barrier refuses it rather than
// forward, so erase-polymorphs fails to legalize the op.

// CHECK: failed to legalize operation 'trait.coerce'
func.func @undischarged(%v: i32, %e: !trait.claim<i32 = i16>) -> i16 {
  %c = trait.coerce %v : i32 to i16 via (%e) : (!trait.claim<i32 = i16>)
  return %c : i16
}
