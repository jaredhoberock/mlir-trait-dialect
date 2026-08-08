// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false %s | FileCheck %s

// erase-polymorphs erases a trait.coerce as a checked judgment, not a trusting
// forward: every cited equality is a claim value that maps to zero at the
// barrier, and so is a claim-typed input. A discharged (reflexive) coerce
// forwards its surviving input; an unused coerce is erased outright even when
// its endpoints differ.

// CHECK-LABEL: func.func @discharged
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.witness
// CHECK: return %arg0 : i1
func.func @discharged(%v: i1) -> i1 {
  %eq = trait.witness refl : !trait.claim<i1 = i1>
  %c = trait.coerce %v : i1 to i1 via (%eq) : (!trait.claim<i1 = i1>)
  return %c : i1
}

// CHECK-LABEL: func.func @unused_nonreflexive
// CHECK-NOT: trait.coerce
// CHECK: return
func.func @unused_nonreflexive(%v: i32, %e: !trait.claim<i32 = i16>) {
  %c = trait.coerce %v : i32 to i16 via (%e) : (!trait.claim<i32 = i16>)
  return
}
