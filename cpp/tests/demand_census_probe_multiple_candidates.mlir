// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The flag classes are orthogonal to the miss arms, so they are pinned against
// a second arm here. Two impls bind @Other[i64], and @Gen's only impl spells
// its own self application through @Other[i64]::X, so matching a candidate for
// @Gen[i64] asks about @Other[i64]::X and gets the candidate-count arm on the
// two-or-more side -- really, speculatively while the resolver partitions
// candidates, and inside the probe.

!T = !trait.poly<0>

trait.trait @Other[!T] {
  trait.assoc_type @X
}

trait.impl @Other_wide for @Other[i64] {
  trait.assoc_type @X = i32
}

trait.impl @Other_narrow for @Other[i64] {
  trait.assoc_type @X = i16
}

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_via for @Gen[!trait.proj<@Other[i64], "X">] {
  trait.assoc_type @A = i32
}

trait.trait @Box[!T] {}

trait.impl @Box_i32 for @Box[i32] {}

func.func private @f(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i32>,
                     %x: !T) -> !T {
  return %x : !T
}

// CHECK: trait-demand-census demand flags=real,speculative,probe-internal drainable=yes observations=8 depth=0
// CHECK-SAME: arms=multiple-candidate-impls
// CHECK-SAME: parent=!trait.proj<@Gen[i64], "A">
// CHECK-SAME: type=!trait.proj<@Other[i64], "X">
// The outer projection the driver's read-only handle declines is the second
// key: its own lookup runs the probe above, and it carries no arm because the
// read never names one.
// CHECK: trait-demand-census demand flags=real drainable=yes observations=1 depth=0
// CHECK-SAME: kinds=read-only-resolver arms=-
// CHECK-SAME: type=!trait.proj<@Gen[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=8 real=2 speculative=2 probe-internal=4
// CHECK: trait-demand-census arm no-candidate-impl keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm multiple-candidate-impls keys=1 observations=8 real=2 speculative=2 probe-internal=4
// CHECK: trait-demand-census summary keys=2 observations=9 drainable-keys=2 unattributed-keys=0 real-keys=2 speculative-keys=0 probe-internal-keys=0
