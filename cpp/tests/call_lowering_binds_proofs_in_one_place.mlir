// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The proofs a call spells are bound where the call is lowered, and nowhere
// else.
//
// Building a call's parameter specialization compares the callee's formal
// signature against the actual one; that comparison needs no proof bindings,
// because the substitution the call is lowered through is closed afterwards and
// the closing walks the same spellings. A specialization that bound them too
// would derive every proof the call names a second time, once for its own map
// and once for the map the call is lowered through.
//
// Both call kinds are lowered here, so both parameter specializations run at
// pass time and neither asks.

!T = !trait.poly<0>

trait.trait @Foo[!T] {
  func.func private @foo(!T) -> !T
}

trait.impl @Foo_impl for @Foo[i64] {
  func.func @foo(%x: i64) -> i64 {
    return %x : i64
  }
}

!U = !trait.poly<1>
func.func nested @through_a_method(%c: !trait.claim<@Foo[!U]>, %x: !U) -> !U {
  %a = trait.assume @Foo[!U]
  %r = trait.method.call %a @Foo[!U]::@foo(%x) : (!U) -> !U
  return %r : !U
}

func.func @main() -> i64 {
  %x = arith.constant 42 : i64
  %c = trait.allege @Foo[i64]
  %r = trait.func.call @through_a_method(%c, %x)
    : (!trait.claim<@Foo[i64]>, i64) -> i64
  return %r : i64
}

// CHECK: trait-demand-census counter call-lowering
// CHECK-SAME: derivations-at-method-call=0
// CHECK-SAME: derivations-at-func-call=0
