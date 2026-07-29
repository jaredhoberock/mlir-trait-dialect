// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// One demanded type asked about eleven times, by two components and under all
// three flag classes, and it produces one entry. That is what a key being the
// demanded type buys: the observation count says how many times the stage
// raised the demand, and the flags say what the askings between them were.
//
// The provenance is the second asker's, not the first's. @probes carries a
// proven claim in its signature, so the declared-proof check asks about
// @Gen[i64]::A before any pattern runs, and matching @Gen_via against it asks
// about @Other[i64]::X inside a candidate probe, at depth 1. The demand @asks
// raises for its own sake comes later, unflagged and at depth 0, and it is that
// asking the entry keeps: a drain reports at the origin the entry names, and a
// probe's location is the wrong place to report.

!T = !trait.poly<0>

trait.trait @Other[!T] {
  trait.assoc_type @X
}

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_via for @Gen[!trait.proj<@Other[i64], "X">] {
  trait.assoc_type @A = i32
}

trait.trait @Box[!T] {}

trait.impl @Box_i32 for @Box[i32] {}

func.func private @probes(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i32>,
                          %x: !T) -> !T {
  return %x : !T
}

func.func @asks() -> !trait.proj<@Other[i64], "X"> {
  // expected-error @below {{unresolved projection '!trait.proj<@Other[i64], "X">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Other[i64], "X">
  return %r : !trait.proj<@Other[i64], "X">
}

// CHECK: trait-demand-census demand flags=real,speculative,probe-internal drainable=yes observations=11 depth=0
// CHECK-SAME: kinds=lookup-miss,resolver-engine-miss arms=no-candidate-impl
// CHECK-SAME: origin=loc({{.*}}demand_census_dedup.mlir":42:1)
// CHECK-SAME: type=!trait.proj<@Other[i64], "X">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=7 real=1 speculative=2 probe-internal=4
// CHECK: trait-demand-census engine resolver-engine-miss keys=1 observations=4 real=4 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=11 drainable-keys=1 unattributed-keys=0 real-keys=1 speculative-keys=0 probe-internal-keys=0
