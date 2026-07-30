// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -stats -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// A call claim that carries no evidence withholds the license to consult module
// facts, so the ground redexes the call's specialization mints are never asked
// about at all. @caller's claim parameter is an ordinary unproven claim and the
// method's signature specializes to @Broad[i64]::Output on both sides, so the
// call's comparison is a spelling match that reads nothing.
//
// Both census channels report on this row, and they report different things.
// The statistic counts the withheld call, because a call op's verifier reaches
// this branch wherever the op is. The ledger's engine column is empty, and one
// precondition is why: method-call lowering, the only in-stage caller of the
// specialization, defers until the call's claim is proven, and a proven claim
// carries the license. So today this engine's population is a verifier's, and
// the statistic is where it is visible. The ledger column becomes nonzero the
// day an in-stage caller reaches the specialization with an unproven claim.

!T = !trait.poly<0>
!X = !trait.proj<@Broad[i64], "Output">

trait.trait @Broad[!T] {
  trait.assoc_type @Output
}

trait.trait @Unwrap[!T] {
  func.func private @unwrap(!T) -> !T
}

// Kept polymorphic so the call op survives to the end of the stage: a monomorph
// carrying an unproven claim is rejected before the method call is reached.
func.func private @caller(%claim: !trait.claim<@Unwrap[!X]>, %value: !X, %spare: !T) -> !X {
  %result = trait.method.call %claim @Unwrap[!X]::@unwrap(%value) : (!X) -> !X
  return %result : !X
}

// CHECK: trait-demand-census engine withheld-call-claim keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=4 drainable-keys=1
// CHECK: 1 trait-demand - calls whose claim withheld the license to consult module facts
