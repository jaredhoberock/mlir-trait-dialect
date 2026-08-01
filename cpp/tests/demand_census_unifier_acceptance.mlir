// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The unifier's acceptances are a population of their own: it found the two
// sides of an equation to be the same projection and returned, having asked no
// impl what that projection resolves to. Nothing here went wrong -- the whole
// module lowers -- and the unifier served nothing all the same, which is why
// the census keeps this arm apart from the lookup's miss arms and never folds
// it into their observation counts.
//
// The key is drainable, but not on the unifier's account. The read-only handle
// the instantiation driver holds declines the same projection, and a read that
// declines leaves the demand standing, so the drain admits the key. One key,
// two engines observing it, so its line names both. The drain serves it in the
// round that collects it -- the whole module lowers with no projection left
// standing.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @Trait[!S, !T] {
  func.func private @method(!S, !T) -> i64
}

trait.trait @Outer[!S] {
  trait.assoc_type @Item
}

func.func private @callee(%t: !T,
    %outer: !trait.claim<@Outer[i64]>,
    %claim: !trait.claim<@Trait[!T, !trait.proj<@Outer[i64], "Item">]>) -> i64 {
  %x = arith.constant 1 : i64
  %px = trait.proj.cast %x, %outer
    : i64 to !trait.proj<@Outer[i64], "Item">
    by !trait.claim<@Outer[i64]>
  %result = trait.method.call %claim @Trait[!T, !trait.proj<@Outer[i64], "Item">]::@method(%t, %px)
    : (!T, !trait.proj<@Outer[i64], "Item">) -> i64
  return %result : i64
}

trait.impl @Outer_i64 for @Outer[i64] {
  trait.assoc_type @Item = i64
}

trait.impl @Trait_i64 for @Trait[i64, i64] {
  func.func @method(%self: i64, %x: i64) -> i64 {
    return %x : i64
  }
}

func.func @main() -> i64 {
  %outer = trait.witness @Outer_i64 for @Outer[i64]
  %trait = trait.witness @Trait_i64 for @Trait[i64, i64]
  %projected = trait.proj.cast %trait, %outer
    : !trait.claim<@Trait[i64, i64] by @Trait_i64>
    to !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>
    by !trait.claim<@Outer[i64] by @Outer_i64>
  %x = arith.constant 0 : i64
  %result = trait.func.call @callee(%x, %outer, %projected)
    : (i64, !trait.claim<@Outer[i64] by @Outer_i64>,
       !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>) -> i64
  return %result : i64
}

// The projection is served in the first round, because the module spells it and
// a round walks the module for what it spells. So the read never meets it and
// the unifier's acceptance is the whole of what this key records -- one
// observation, and not one a later round could be asked to serve.
// CHECK: trait-stage-record round index=1
// CHECK-SAME: collected=2
// CHECK-SAME: served=2
// CHECK: trait-demand-census demand flags=real drainable=no observations=1 depth=0
// CHECK-SAME: kinds=unifier-acceptance
// CHECK-SAME: type=!trait.proj<@Outer[i64], "Item">
// CHECK: trait-demand-census engine lookup-miss keys=0 observations=0 real=0 speculative=0 probe-internal=0
// CHECK: trait-demand-census engine unifier-acceptance keys=1 observations=1 real=1 speculative=0 probe-internal=0
// CHECK: trait-demand-census summary keys=1 observations=1 drainable-keys=0
