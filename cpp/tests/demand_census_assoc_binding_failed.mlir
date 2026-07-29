// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// One impl binds @Gen[i64], but the projection names an associated type the
// trait never declared, so the impl has no binding to read. The census keeps
// this apart from the candidate-count arms: the impl was found, and the demand
// still went unserved.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_i64 for @Gen[i64] {
  trait.assoc_type @A = i32
}

func.func @wrap(%x: !T) -> !trait.proj<@Gen[!T], "B"> {
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "B">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Gen[!T], "B">
  return %r : !trait.proj<@Gen[!T], "B">
}

func.func @main() -> !trait.proj<@Gen[i64], "B"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "B">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "B">
  return %r : !trait.proj<@Gen[i64], "B">
}

// CHECK: trait-demand-census demand flags=real drainable=yes observations=15 depth=0
// CHECK-SAME: arms=assoc-binding-failed
// CHECK-SAME: type=!trait.proj<@Gen[i64], "B">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm assoc-binding-failed keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=15 drainable-keys=1
