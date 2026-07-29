// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// Two impls bind @Gen[i64], so choosing between them is a premise partition the
// read-only lookup does not perform. The census tells that apart from having no
// impl at all: the two are separate arms, and only one of them is a projection
// waiting on a generator.

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
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Gen[!T], "A">
  return %r : !trait.proj<@Gen[!T], "A">
}

func.func @main() -> !trait.proj<@Gen[i64], "A"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "A">
  return %r : !trait.proj<@Gen[i64], "A">
}

// CHECK: trait-demand-census demand flags=real drainable=yes observations=15 depth=0
// CHECK-SAME: arms=multiple-candidate-impls
// CHECK-SAME: type=!trait.proj<@Gen[i64], "A">
// CHECK: trait-demand-census engine lookup-miss keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm no-candidate-impl keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm multiple-candidate-impls keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=15 drainable-keys=1
