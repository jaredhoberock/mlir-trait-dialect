// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Tests that impl GAT arity mismatches are diagnosed.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @HasGAT(%self: !trait.claim<@HasGAT[!S]>) {
  trait.assoc_type @Item<[!T]>
  trait.method @get(!S, !T) -> !trait.proj<@HasGAT[!S], "Item", [!T]>
}

// expected-error @+1 {{'trait.impl' op associated type 'Item' has 0 type parameter(s) but trait declares 1}}
trait.impl private @HasGAT_impl(%self_claim: !trait.claim<@HasGAT[i32]>) {
  trait.assoc_type @Item = i64
  trait.method @get(%self: i32, %value: i64) -> i64 {
    %c = arith.constant 42 : i64
    trait.return %c : i64
  }
}

// -----

!S = !trait.poly<0>
!T = !trait.poly<1>
!U = !trait.poly<2>

trait.trait private @OneParam(%self: !trait.claim<@OneParam[!S]>) {
  trait.assoc_type @Item<[!T]>
  trait.method @get(!S, !T) -> !trait.proj<@OneParam[!S], "Item", [!T]>
}

// expected-error @+1 {{'trait.impl' op associated type 'Item' has 2 type parameter(s) but trait declares 1}}
trait.impl private @OneParam_impl(%self_claim: !trait.claim<@OneParam[i32]>) {
  trait.assoc_type @Item<[!T, !U]> = !T
  trait.method @get(%self: i32, %value: i64) -> i64 {
    %c = arith.extsi %self : i32 to i64
    trait.return %c : i64
  }
}
