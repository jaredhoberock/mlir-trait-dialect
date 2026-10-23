// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false %s | FileCheck %s

// erase-polymorphs erases a trait.coerce by its result's kind: every cited
// equality is a claim value that maps to zero at the barrier, and so is a
// claim-typed input. A discharged (reflexive) coerce of a value forwards its
// surviving input; a coerce to a claim erases 1:0 on the judgment the pass makes
// at its entry, which refuses a claim nothing proves before anything is erased
// (see invalid_coerce_of_undischarged_evidence_is_refused_at_erasure).

// CHECK-LABEL: func.func @discharged
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.witness
// CHECK: return %arg0 : i1
func.func @discharged(%v: i1) -> i1 {
  %eq = trait.witness refl : !trait.claim<i1 = i1>
  %c = trait.coerce %v : i1 to i1 via (%eq) : (!trait.claim<i1 = i1>)
  return %c : i1
}

// CHECK-LABEL: func.func @dead_reflexive_respell
// CHECK-NOT: trait.
// CHECK-NEXT: return
func.func @dead_reflexive_respell() {
  %b = trait.witness @Bound_i64 for @Bound[i64]
  %c = trait.coerce %b : !trait.claim<@Bound[i64] by @Bound_i64> to !trait.claim<@Bound[i64] by @Bound_i64>
  return
}

trait.trait private @Bound(%self: !trait.claim<@Bound[!trait.poly<0>]>) {}
trait.impl private @Bound_i64(%self: !trait.claim<@Bound[i64]>) {}
