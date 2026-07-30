// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Each greedy pattern driver the stage runs reports the rewrite events it
// raised, so a change that moves work between drivers is visible as the counts
// moving with it. The module is one allegation and one method call: nothing
// converts, the allegation becomes a witness under resolve-impls, and
// instantiate-monomorphs specializes the callee and lowers the call.
//
// The budget is what the driver stops at, so the headroom beside it says how
// close a run came to the bound that catches a non-confluent pattern pair. Only
// instantiate-monomorphs carries one.

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

// CHECK: trait-stage-record rewrites driver=convert-to-trait inserted=0 modified=0 replaced=0 erased=0 applications=0 budget=unbounded headroom=unbounded
// CHECK: trait-stage-record rewrites driver=resolve-impls inserted=1 modified=1 replaced=1 erased=1 applications=1 budget=unbounded headroom=unbounded
// CHECK: trait-stage-record rewrites driver=instantiate-monomorphs inserted=4 modified=3 replaced=2 erased=2 applications=2 budget=14336 headroom=14334
