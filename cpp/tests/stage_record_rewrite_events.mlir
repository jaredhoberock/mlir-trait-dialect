// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Each greedy pattern driver the stage runs reports the rewrite events it
// raised and the round it ran in, so a change that moves work between drivers
// or between rounds is visible as the counts moving with it. The module is one
// allegation and one method call: nothing converts, the allegation becomes a
// witness in round zero, and the first round specializes the callee and lowers
// the call.
//
// The budget is what the driver stops at, so the headroom beside it says how
// close a run came to the bound that catches a non-confluent pattern pair.
//
// Rounds run until one of them writes nothing, so the last round of every run
// is one that found no demand to serve and respelled nothing. That is what its
// line says, and it is why there is one more round here than there is work to
// do.
//
// A step whose input has not moved since it last ran does not run again, which
// is why round one reports no respelling sweep: round zero swept after its own
// writes and the bridge that ran before the sweep would have moved nothing. The
// same holds of the instantiation driver, so the last round reports no rewrite
// events for it at all: nothing since that driver's own last run moved the
// module it reads or the facts it reads them against.

!T = !trait.poly<0>

trait.trait @Greet[!T] {
  func.func private @greet(!T) -> i32
}

trait.impl @Greet_i32 for @Greet[i32] {
  func.func @greet(%x: i32) -> i32 {
    return %x : i32
  }
}

func.func @main(%x: i32) -> i32 {
  %w = trait.allege @Greet[i32]
  %r = trait.method.call %w @Greet[i32]::@greet(%x) : (i32) -> i32
  return %r : i32
}

// CHECK: trait-stage-record rewrites driver=convert-to-trait round=0 inserted=0 modified=0 replaced=0 erased=0 applications=0 budget=14336 headroom=14336
// CHECK: trait-stage-record rewrites driver=resolve-impls round=0 inserted=1 modified=1 replaced=1 erased=1 applications=1 budget=unbounded headroom=unbounded
// CHECK: trait-stage-record respelling round=0 bindings=1 ops=0 positions=0
// CHECK: trait-stage-record rewrites driver=convert-to-trait round=1 inserted=0 modified=0 replaced=0 erased=0 applications=0 budget=14336 headroom=14336
// CHECK-NOT: trait-stage-record respelling round=1
// CHECK: trait-stage-record rewrites driver=instantiate-monomorphs round=1 inserted=4 modified=3 replaced=2 erased=2 applications=2 budget=14336 headroom=14334
// CHECK: trait-stage-record round index=1 bridged=no collected=0
// CHECK-SAME: served=0 declined=0 deferred=0
// CHECK-SAME: instantiated=yes
// CHECK: trait-stage-record respelling round=2 bindings=1 ops=0 positions=0
// CHECK-NOT: trait-stage-record rewrites driver=instantiate-monomorphs round=2
// CHECK: trait-stage-record round index=2 bridged=no collected=0
// CHECK-SAME: ambiguous-arms=0 served=0 declined=0 deferred=0 inserted-serving-demands=0
// CHECK-SAME: respelled-positions=0 refusals-forgotten=0 refusals-kept=0 refusals-overturned=0 refusals-re-earned=0 instantiate-minted=0 proven-producers=0 instantiated=no
