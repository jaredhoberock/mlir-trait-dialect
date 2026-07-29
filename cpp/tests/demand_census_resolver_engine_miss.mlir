// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The resolver's own projection engine is a second way a projection is asked
// about, and its callers keep the projection spelled as written when it fails
// rather than reporting anything. The census is where that failure becomes
// visible, in its own arm: nothing here consulted the read-only lookup.

trait.trait @T[!trait.poly<0>] {
  trait.assoc_type @A
}

func.func @main() -> !trait.proj<@T[i64], "A"> {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@T[i64], "A">
  return %r : !trait.proj<@T[i64], "A">
}

// CHECK: trait-demand-census demand flags=real drainable=yes observations=2 depth=0
// CHECK-SAME: kinds=resolver-engine-miss
// CHECK-SAME: type=!trait.proj<@T[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine resolver-engine-miss keys=1 observations=2 real=2 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=2 drainable-keys=1
