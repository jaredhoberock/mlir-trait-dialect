// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// One impl binds @Gen[i64], but the projection names an associated type the
// trait never declared, so the impl has no binding to read. The census keeps
// this apart from the candidate-count arms: the impl was found, and the demand
// still went unserved.
//
// @probes carries a proven claim spelled through that projection, so the
// declared-proof check normalizes it through the read-only lookup, which is the
// one engine that names the arm it missed on, and then verifies the proof's
// cited impl against the claim, driving that same ground projection through one
// more lookup that misses. The round then takes the demand off the drain and
// puts it to impl selection, which settles on @Gen_i64 and still has no binding
// to answer with, so the demand is deferred with an impl selected and no refusal
// recorded.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_i64 for @Gen[i64] {
  trait.assoc_type @A = i32
}

trait.trait @Box[!T] {}

trait.impl @Box_i32 for @Box[i32] {}

func.func private @probes(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "B">] by @Box_i32>,
                          %x: !T) -> !T {
  return %x : !T
}

func.func @wrap(%x: !T) -> !trait.proj<@Gen[!T], "B"> {
  %r = ub.poison : !trait.proj<@Gen[!T], "B">
  return %r : !trait.proj<@Gen[!T], "B">
}

func.func @main() -> !trait.proj<@Gen[i64], "B"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "B">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "B">
  return %r : !trait.proj<@Gen[i64], "B">
}

// CHECK: trait-stage-record round index=1
// CHECK-SAME: collected=1 no-candidate-impl=0 multiple-candidate-impls=0 other-arms=1 without-arm=0
// CHECK-SAME: served=0 declined=1 deferred=1
// CHECK: trait-demand-census demand flags=real drainable=yes observations=9 depth=0
// CHECK-SAME: kinds=lookup-miss,unifier-acceptance,read-only-resolver arms=assoc-binding-failed
// CHECK-SAME: type=!trait.proj<@Gen[i64], "B">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=5 real=5 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine read-only-resolver keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm assoc-binding-failed keys=1 observations=5 real=5 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=9 drainable-keys=1
// CHECK: trait-stage-record digest value={{.*}} selected-impls=1 refusals-no-candidate=0 refusals-ambiguous=0
