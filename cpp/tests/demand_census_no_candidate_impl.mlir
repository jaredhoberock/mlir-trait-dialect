// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// @Gen has no impl at all, so specializing @wrap for i64 mints
// @Gen[i64]::A -- a projection whose trait application no impl binds. The
// lookup declines on the no-candidate arm and the census records the demand
// once, however many components asked about it: the key is the demanded type,
// so repeat askings merge into one line whose observation count is the number
// of times the stage raised it.
//
// The two arms below the recorded one are pinned at zero because neither is
// reachable from a module that verifies. A projection over a trait the module
// does not define is rejected by symbol-use verification before any pass runs,
// and a self-claim substitution failure cannot survive candidate enumeration,
// which admits a candidate only by running that very substitution.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
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

// CHECK: trait-demand-census demand flags=real drainable=yes observations=18 depth=0
// CHECK-SAME: arms=no-candidate-impl
// CHECK-SAME: type=!trait.proj<@Gen[i64], "A">
// CHECK: trait-demand-census arm trait-symbol-not-found keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm no-candidate-impl keys=1 observations=3 real=3 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm self-claim-substitution-failed keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=18 drainable-keys=1
