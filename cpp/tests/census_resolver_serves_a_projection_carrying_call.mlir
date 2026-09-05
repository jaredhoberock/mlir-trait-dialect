// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A method call whose receiver claim spells a projection over the impl's
// associated type (@Outer[i64]::Item). The resolver resolves that projection
// through @Outer_i64 and serves both sides, so the call lowers with the two
// demands served and the ledger empty -- the unifier never reaches its
// projection-crossing acceptance here, because the candidate resolves the
// crossing first. (That acceptance path is exercised across the corpus; this
// row pins the resolver-served route a projection-carrying call now takes.)
//
// RUN: env TRAIT_DEMAND_CENSUS=1 TRAIT_DEMAND_CENSUS_CHECK=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --implicit-check-not='trait-demand-census unhooked' --implicit-check-not='trait-demand-census served'

// The unifier's acceptances are a population of their own: it found the two
// sides of an equation to be the same projection and returned, having asked no
// impl what that projection resolves to. Nothing here went wrong -- the whole
// module lowers -- and the unifier served nothing all the same, which is why
// the census keeps this arm apart from the lookup's miss arms and never folds
// it into their observation counts.
//
// The key is not drainable, and the round is why. A round collects what the
// module spells, so this projection is put to impl selection in the round that
// finds it and served there; the read-only handle the instantiation driver
// holds never meets it, nothing declines it, and the drain has no key to admit.
// One engine observes it, so its line names one -- and the acceptance is not an
// observation a later round could be asked to serve.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Trait[!S, !T] {
  func.func private @method(!S, !T) -> i64
}

trait.trait private @Outer[!S] {
  trait.assoc_type @Item
}

func.func private @callee(%t: !T,
    %outer: !trait.claim<@Outer[i64]>,
    %claim: !trait.claim<@Trait[!T, !trait.proj<@Outer[i64], "Item">]>) -> i64 {
  %x = arith.constant 1 : i64
  %px = trait.coerce %x : i64 to !trait.proj<@Outer[i64], "Item"> unproven
  %result = trait.method.call %claim @Trait[!T, !trait.proj<@Outer[i64], "Item">]::@method(%t, %px)
    : (!T, !trait.proj<@Outer[i64], "Item">) -> i64
  return %result : i64
}

trait.impl private @Outer_i64 for @Outer[i64] {
  trait.assoc_type @Item = i64
}

trait.impl private @Trait_i64 for @Trait[i64, i64] {
  func.func @method(%self: i64, %x: i64) -> i64 {
    return %x : i64
  }
}

func.func @main() -> i64 {
  %outer = trait.witness @Outer_i64 for @Outer[i64]
  %trait = trait.witness @Trait_i64 for @Trait[i64, i64]
  %eq = trait.witness proj_resolve !trait.proj<@Outer[i64], "Item"> resolves i64 by @Outer_i64
    : !trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>
  %projected = trait.coerce %trait
    : !trait.claim<@Trait[i64, i64] by @Trait_i64>
    to !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>
    via (%eq) : (!trait.claim<!trait.proj<@Outer[i64], "Item"> = i64>)
  %x = arith.constant 0 : i64
  %result = trait.func.call @callee(%x, %outer, %projected)
    : (i64, !trait.claim<@Outer[i64] by @Outer_i64>,
       !trait.claim<@Trait[i64, !trait.proj<@Outer[i64], "Item">] by @Trait_i64>) -> i64
  return %result : i64
}

// CHECK: trait-stage-record round index=1
// CHECK-SAME: collected=2
// CHECK-SAME: served=2
// The two projection sides now unify through a candidate before the unifier
// reaches its accept-irreducible path, so no acceptance is recorded and the
// census is empty; the module still lowers (served=2 above).
// CHECK: trait-demand-census summary keys=0 observations=0 drainable-keys=0
