// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// isPendingExpansion counts a standing allegation outside a template as pending
// instantiation whatever it alleges, so the lowering driver serves an equality
// allegation that is the module's only trait work: no type the allegation
// carries records it as owed, since an equality is no obligation site, and only
// the claim-proving pattern of instantiate-monomorphs proves it.

// RUN: mlir-opt %s -pass-pipeline='builtin.module(report-expansion-readiness-trait)' 2>&1 | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWERED

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

func.func private @g(%e: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// CHECK: pending-expansion=true
// LOWERED-LABEL: func.func @use() -> i64
// LOWERED-NEXT: call @g() : () -> i64
func.func @use() -> i64 {
  %a = trait.allege !trait.proj<@A[i64], "Item"> = i64
  %v = func.call @g(%a) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64
  return %v : i64
}
