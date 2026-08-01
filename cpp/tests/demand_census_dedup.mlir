// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// One demanded type asked about fourteen times, by two components and under all
// three flag classes, and it produces one entry. That is what a key being the
// demanded type buys: the observation count says how many times the stage
// raised the demand, and the flags say what the askings between them were.
//
// The second component is the read of impl selection's recorded facts the
// instantiation driver's patterns hold: what it has no answer for it leaves
// spelled as written, for the round that may make selection answer it. That
// read also meets @Gen[i64]::A on its way to @probes' claim, which is the
// second entry: a different demanded type is a different key, however few times
// it is asked about.
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

// CHECK: trait-demand-census demand flags=real,speculative,probe-internal drainable=yes observations=14 depth=0
// CHECK-SAME: kinds=lookup-miss,read-only-resolver arms=no-candidate-impl
// CHECK-SAME: origin=loc({{.*}}demand_census_dedup.mlir":49:1)
// CHECK-SAME: type=!trait.proj<@Other[i64], "X">
// CHECK: trait-demand-census demand flags=real drainable=yes observations=1 depth=0
// CHECK-SAME: kinds=read-only-resolver arms=-
// CHECK-SAME: origin=loc({{.*}}demand_census_dedup.mlir":44:1)
// CHECK-SAME: type=!trait.proj<@Gen[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=8 real=2 speculative=2 probe-internal=4
// CHECK: trait-demand-census engine resolver-engine-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine read-only-resolver keys=2 observations=7 real=7 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=2 observations=15 drainable-keys=2 unattributed-keys=0 real-keys=2 speculative-keys=0 probe-internal-keys=0
