// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// One demanded type asked about twelve times, by two components and under all
// three flag classes, and it produces one entry. That is what a key being the
// demanded type buys: the observation count says how many times the stage
// raised the demand, and the flags say what the askings between them were.
//
// The second component is the read of impl selection's recorded facts the
// instantiation driver's patterns hold: what it has no answer for it leaves
// spelled as written, for the round that may make selection answer it. Four of
// the twelve askings are that read declining @Other[i64]::X, which no impl of
// @Other binds; the other eight are impl selection's own lookup, half of them
// raised inside the candidate probe that matches @Gen_via.
//
// @Gen[i64]::A is no key of this census at all. A round collects what the
// module spells, so it is put to impl selection in round one and served there;
// the read the driver holds never meets it, and what deduplicates here is the
// one key that serve raised while it ran.
//
// The provenance is the first real asker's, and a probe is not one. A key keeps
// the first asking that is real, outside a candidate probe and under a frame:
// that asking is raised while @Gen[i64]::A is being resolved -- the projection
// @probes' signature spells -- so the entry names that projection as its parent
// and reports at @probes' declaration. The probe-internal askings never take
// the provenance, because a drain reports at the origin the entry names and a
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

// CHECK: trait-demand-census demand flags=real,speculative,probe-internal drainable=yes observations=12 depth=0
// CHECK-SAME: kinds=lookup-miss,read-only-resolver arms=no-candidate-impl
// CHECK-SAME: origin=loc({{.*}}demand_census_dedup.mlir":49:1)
// CHECK-SAME: type=!trait.proj<@Other[i64], "X">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=8 real=2 speculative=2 probe-internal=4
// CHECK: trait-demand-census engine resolver-engine-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine read-only-resolver keys=1 observations=4 real=4 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=12 drainable-keys=1 unattributed-keys=0 real-keys=1 speculative-keys=0 probe-internal-keys=0
