// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The instantiation driver reads what impl selection has recorded and never
// makes it run, so a demand the record does not answer is one it leaves spelled
// as written for a step that can. The census is where that decline becomes
// visible, in its own engine and under no arm: naming which way an application
// missed is selection's, and this read never ran it.

trait.trait @T[!trait.poly<0>] {
  trait.assoc_type @A
}

func.func @main() -> !trait.proj<@T[i64], "A"> {
  // expected-error @below {{unresolved projection '!trait.proj<@T[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@T[i64], "A">
  return %r : !trait.proj<@T[i64], "A">
}

// CHECK: trait-demand-census demand flags=real drainable=yes observations=4 depth=0
// CHECK-SAME: kinds=read-only-resolver arms=-
// CHECK-SAME: type=!trait.proj<@T[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine resolver-engine-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine read-only-resolver keys=1 observations=4 real=4 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=4 drainable-keys=1
