// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The resolver's own projection engine is a second way a projection is asked
// about. Impl selection resolves the projections spelled inside a demanded
// claim before it selects, and where one of them will not resolve, selection
// carries on with the projection spelled as written and reports nothing. The
// census is where that failure becomes visible, in its own arm.
//
// @T has no impl, so @T[i64]::A never resolves, while the blanket @Box_any
// binds the claim spelled through it, so selection settles the claim and leaves
// the projection standing. The read the instantiation driver holds meets the
// same projection and is counted apart from selection's own engine, which is
// what tells a resolution that failed from a read that had no answer.

!T = !trait.poly<0>

trait.trait private @T[!T] {
  trait.assoc_type @A
}

trait.trait private @Box[!T] {}

trait.impl private @Box_any for @Box[!T] {}

func.func @callee(%c: !trait.claim<@Box[!trait.proj<@T[i64], "A">]>,
                  %x: !T) -> !T {
  return %x : !T
}

func.func @main() -> i64 {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %c = trait.allege @Box[!trait.proj<@T[i64], "A">]
  %x = arith.constant 0 : i64
  %r = trait.func.call @callee(%c, %x)
    : (!trait.claim<@Box[!trait.proj<@T[i64], "A">]>, i64) -> i64
  return %r : i64
}

// CHECK: trait-demand-census demand flags=real drainable=yes observations=9 depth=0
// CHECK-SAME: kinds=lookup-miss,unifier-acceptance,resolver-engine-miss,read-only-resolver arms=no-candidate-impl
// CHECK-SAME: type=!trait.proj<@T[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=4 real=4 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine unifier-acceptance keys=1 observations=1 real=1 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine resolver-engine-miss keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine read-only-resolver keys=1 observations=1 real=1 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=9 drainable-keys=1
// CHECK: trait-demand-census counter total residual-tolerance-accepts=0 residual-tolerance-accepts-generator-pending=0 residual-tolerance-accepts-multi-candidate=0 residual-tolerance-accepts-hypothesis=0 residual-tolerance-accepts-mixed-or-other=0 resolver-engine-misses=3
