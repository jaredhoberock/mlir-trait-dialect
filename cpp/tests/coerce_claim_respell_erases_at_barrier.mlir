// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' --verify-each=false | FileCheck %s

// The polymorph-erasing barrier maps a DISCHARGED claim-respell coerce to
// nothing: once its projections have ground and its endpoints coincide, its
// claim-typed operand and its cited equality are proof material that carries no
// runtime value, so the 1:0 erasure empties the function body. Dropping it is
// conditioned on the endpoints matching -- a respell whose projection never
// ground still relates two spellings and is refused, not erased (see
// invalid_coerce_undischarged_at_barrier).

trait.trait @Bound[!trait.poly<0>] {
}

// CHECK-LABEL: func.func @discharged_claim_respell
// CHECK-NOT: trait.coerce
// CHECK-NEXT: return
func.func @discharged_claim_respell(%b: !trait.claim<@Bound[i64]>,
                                    %eq: !trait.claim<i64 = i64>) {
  %c = trait.coerce %b : !trait.claim<@Bound[i64]>
    to !trait.claim<@Bound[i64]>
    via (%eq) : (!trait.claim<i64 = i64>)
  return
}
