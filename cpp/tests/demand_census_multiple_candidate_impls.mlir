// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// Two impls bind @Gen[i64], so the read of impl selection's recorded facts that
// the instantiation driver holds has no answer for @Gen[i64]::A and leaves it
// spelled as written, exactly as it would for a trait with no impl at all. What
// tells the two apart is the round: it puts the demand to selection, and two
// satisfiable candidates is the refusal no later fact overturns -- candidates
// are only ever appended -- so the demand leaves the drain refused where a
// missing impl would have left it deferred.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_wide for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.impl @Gen_narrow for @Gen[i64] {
  trait.assoc_type @A = i16
}

func.func @wrap(%x: !T) -> !trait.proj<@Gen[!T], "A"> {
  %r = ub.poison : !trait.proj<@Gen[!T], "A">
  return %r : !trait.proj<@Gen[!T], "A">
}

func.func @main() -> !trait.proj<@Gen[i64], "A"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "A">
  return %r : !trait.proj<@Gen[i64], "A">
}

// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=1 no-candidate-impl=0 multiple-candidate-impls=0 other-arms=0 without-arm=1
// CHECK-SAME: served=0 declined=1 deferred=0
// CHECK: trait-demand-census demand flags=real drainable=yes observations=10 depth=0
// CHECK-SAME: kinds=unifier-acceptance,read-only-resolver arms=-
// CHECK-SAME: type=!trait.proj<@Gen[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine read-only-resolver keys=1 observations=8 real=8 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm no-candidate-impl keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm multiple-candidate-impls keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=10 drainable-keys=1
// CHECK: trait-stage-record digest value={{.*}} selected-impls=0 refusals-no-candidate=0 refusals-ambiguous=1
