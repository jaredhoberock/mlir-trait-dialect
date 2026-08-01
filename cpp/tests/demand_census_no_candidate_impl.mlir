// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// @Gen has no impl at all, so the read of impl selection's recorded facts that
// the instantiation driver holds has no answer for @Gen[i64]::A and leaves it
// spelled as written. The census records the demand once, however many
// components asked about it: the key is the demanded type, so repeat askings
// merge into one line whose observation count is the number of times the stage
// raised it.
//
// The first round collects it, because the module spells it and a round walks
// the module for what it spells. Nothing has asked the lookup about it at that
// point, so it arrives under the arm-less column; the round puts it to
// selection, which finds no candidate, and that is the refusal an impl
// generated later would overturn, so the demand is deferred rather than left
// drained. The arms the census records against it are the lookup's, raised
// afterwards while the call that spells it was being lowered.

!T = !trait.poly<0>

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

func.func @wrap(%x: !T) -> !trait.proj<@Gen[!T], "A"> {
  %r = ub.poison : !trait.proj<@Gen[!T], "A">
  return %r : !trait.proj<@Gen[!T], "A">
}

func.func @main() -> !trait.proj<@Gen[i64], "A"> {
  %x = arith.constant 1 : i64
  // expected-error @below {{unresolved projection '!trait.proj<@Gen[i64], "A">' after instantiate-monomorphs}}
  %r = trait.func.call @wrap(%x) : (i64) -> !trait.proj<@Gen[i64], "A">
  return %r : !trait.proj<@Gen[i64], "A">
}

// CHECK: trait-stage-record round index=1
// CHECK-SAME: collected=1 no-candidate-impl=0 multiple-candidate-impls=0 other-arms=0 without-arm=1
// CHECK-SAME: served=0 declined=1 deferred=1
// CHECK: trait-demand-census demand flags=real drainable=yes observations=5 depth=0
// CHECK-SAME: kinds=lookup-miss,unifier-acceptance,read-only-resolver arms=no-candidate-impl
// CHECK-SAME: type=!trait.proj<@Gen[i64], "A">
// CHECK: trait-demand-census engine read-only-resolver keys=1 observations=2 real=2 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm trait-symbol-not-found keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm no-candidate-impl keys=1 observations=2 real=2 speculative=0 probe-internal=0
// CHECK: trait-demand-census arm self-claim-substitution-failed keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=5 drainable-keys=1
// The refusal itself does not survive to the digest: it is the arm a later
// impl could overturn, so the next round's flush drops it, and the stage ends
// with the demand standing and no fact recorded about it.
// CHECK: trait-stage-record digest value={{.*}} selected-impls=0 refusals-no-candidate=0 refusals-ambiguous=0
