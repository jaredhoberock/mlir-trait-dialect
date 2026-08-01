// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics 2>&1 | FileCheck %s

// A round whose only work was to forget a refusal runs no instantiation driver.
//
// The flush leads a round so that the questions after it are asked against the
// facts as they now stand, and dropping a negative is what makes impl selection
// derive that application again. It is not something the driver reads: a read of
// the record fails on a refused application exactly as it fails on one selection
// has never been asked about, so the driver would be handed the module its own
// last run left, together with the record that run read.
//
// @Gen[i64]::A resolves once @Gen_via is selected, which mints a proof and keeps
// the loop going; @Other[i64]::X never resolves, and the refusal selection
// records for it is the one the next round's flush drops. That round finds
// nothing else, and its line says so.

!T = !trait.poly<0>

trait.trait @Other[!T] {
  trait.assoc_type @X
}

trait.trait @Gen[!T] {
  trait.assoc_type @A
}

trait.impl @Gen_via for @Gen[!trait.proj<@Other[i64], "X">] {
  trait.assoc_type @A = i32
}

trait.trait @Box[!T] {}

trait.impl @Box_i32 for @Box[i32] {}

func.func private @reads(%c: !trait.claim<@Box[!trait.proj<@Gen[i64], "A">] by @Box_i32>,
                         %x: !T) -> !T {
  return %x : !T
}

func.func @asks() -> !trait.proj<@Other[i64], "X"> {
  // expected-error @below {{unresolved projection '!trait.proj<@Other[i64], "X">' after instantiate-monomorphs}}
  %r = ub.poison : !trait.proj<@Other[i64], "X">
  return %r : !trait.proj<@Other[i64], "X">
}

// CHECK: trait-stage-record round index=1
// CHECK-SAME: served=1
// CHECK-SAME: instantiated=yes
// CHECK-NOT: trait-stage-record rewrites driver=instantiate-monomorphs round=2
// CHECK: trait-stage-record round index=2
// CHECK-SAME: collected=0
// CHECK-SAME: served=0 declined=0 deferred=0 inserted-serving-demands=0
// CHECK-SAME: respelled-positions=0 refusals-forgotten=1
// CHECK-SAME: instantiated=no
