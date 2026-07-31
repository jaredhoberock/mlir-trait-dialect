// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The only impl of @Gen spells its own self application with a projection, so
// matching a candidate against @Gen[i64] resolves @Other[i64]::X on the way --
// and @Other has no impl. That nested lookup runs inside the outer lookup's
// replacement callback, which is what makes it a candidate probe rather than a
// demand of its own.
//
// One key collects all three flag classes here, which is the point of the row:
// the same demanded type is raised really, speculatively while the resolver
// partitions candidates, and inside a probe. Only the unflagged observation
// makes the key drainable, and the parent it names is the demand under
// resolution rather than the candidate the probe was about.

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

func.func private @f(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i32>,
                     %x: !T) -> !T {
  return %x : !T
}

// CHECK: trait-demand-census demand flags=real,speculative,probe-internal drainable=yes observations=8 depth=0
// CHECK-SAME: arms=no-candidate-impl
// CHECK-SAME: parent=!trait.proj<@Gen[i64], "A">
// CHECK-SAME: type=!trait.proj<@Other[i64], "X">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=8 real=2 speculative=2 probe-internal=4
// CHECK: trait-demand-census arm no-candidate-impl keys=1 observations=8 real=2 speculative=2 probe-internal=4
// CHECK: trait-demand-census summary keys=2 observations=9 drainable-keys=2 unattributed-keys=0
