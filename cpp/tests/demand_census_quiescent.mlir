// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// Every projection here is served where it is raised, so the stage finishes
// with an empty population. This is the negative pin the other rows are read
// against: a census that reports demand on this module is reporting something
// the stage did not have.

!T = !trait.poly<0>

trait.trait @Outer[!T] {
  trait.assoc_type @Item
}

trait.impl @Outer_i64 for @Outer[i64] {
  trait.assoc_type @Item = i64
}

trait.trait @Sink[!T] {}

trait.impl @Sink_any for @Sink[!T] {}

func.func @callee(%c: !trait.claim<@Sink[!trait.proj<@Outer[i64], "Item">]>,
                  %x: !T) -> !T {
  return %x : !T
}

func.func @main() -> i64 {
  %c = trait.allege @Sink[!trait.proj<@Outer[i64], "Item">]
  %x = arith.constant 0 : i64
  %r = trait.func.call @callee(%c, %x)
    : (!trait.claim<@Sink[!trait.proj<@Outer[i64], "Item">]>, i64) -> i64
  return %r : i64
}

// CHECK-NOT: trait-demand-census demand
// CHECK: trait-demand-census engine lookup-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=0 observations=0 drainable-keys=0
