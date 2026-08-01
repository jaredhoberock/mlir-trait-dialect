// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// A round collects what the module spells, not only what a component wrote down.
//
// `@holds` is a template, and the claim it takes is monomorphic where its own
// type parameter is not. The proving rule keys on the result a claim producer
// holds, so it never looks at a block argument; nothing calls this function, so
// no call site specializes it; and no engine is asked about the claim, so the
// ledger records no demand for it. Before the round walked the module there was
// nothing to collect, and impl selection was never asked -- the claim stood at
// the end of the stage with an impl in the module that proves it.
//
// The walk finds it where it is spelled, the round puts it to selection, and
// the proof is recorded.

!T = !trait.poly<0>

trait.trait @P[!T] {}

trait.impl @P_i64 for @P[i64] {}

func.func nested @holds(%c: !trait.claim<@P[i64]>, %x: !T) -> !T {
  return %x : !T
}

// CHECK: trait-stage-record round index=1
// CHECK-SAME: collected=1
// CHECK-SAME: without-arm=1
// CHECK-SAME: served=1
// CHECK: trait-stage-record digest
// CHECK-SAME: selected-impls=1
