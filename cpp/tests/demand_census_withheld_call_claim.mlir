// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -stats -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// A method call whose claim carries no evidence withholds the license to consult
// module facts. @caller's claim parameter is an ordinary unproven claim, so the
// call's verifier compares its formal and actual with the module-free comparator
// and reads nothing. The demand this raises is a verifier's -- counted by the
// statistic, never entered in the ledger: no stage raises it, and no ledger
// engine hears it, so the census summary is empty while the statistic is one.

!T = !trait.poly<0>

trait.trait private @Unwrap[!T] {
  func.func private @unwrap(!T) -> !T
}

// Kept polymorphic so the call op survives to the end of the stage: a monomorph
// carrying an unproven claim is rejected before the method call is reached.
func.func private @caller(%claim: !trait.claim<@Unwrap[!T]>, %value: !T) -> !T {
  %result = trait.method.call %claim @Unwrap[!T]::@unwrap(%value) : (!T) -> !T
  return %result : !T
}

// CHECK-NOT: trait-demand-census engine withheld-call-claim
// CHECK: trait-demand-census summary keys=0 observations=0 drainable-keys=0
// CHECK: 1 trait-demand - calls whose claim withheld the license to consult module facts
