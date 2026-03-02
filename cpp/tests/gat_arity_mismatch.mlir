// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// Tests that impl GAT arity mismatches are diagnosed.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @HasGAT[!S] {
  trait.assoc_type @Item<[!T]>
  func.func private @get(!S) -> !trait.proj<@HasGAT[!S], "Item", [!T]>
}

// expected-error @+1 {{'trait.impl' op associated type 'Item' has 0 type parameter(s) but trait declares 1}}
trait.impl for @HasGAT[i32] {
  trait.assoc_type @Item = i64
  func.func @get(%self: i32) -> i64 {
    %c = arith.constant 42 : i64
    return %c : i64
  }
}

// -----

!S = !trait.poly<0>
!T = !trait.poly<1>
!U = !trait.poly<2>

trait.trait @OneParam[!S] {
  trait.assoc_type @Item<[!T]>
  func.func private @get(!S) -> !trait.proj<@OneParam[!S], "Item", [!T]>
}

// expected-error @+1 {{'trait.impl' op associated type 'Item' has 2 type parameter(s) but trait declares 1}}
trait.impl for @OneParam[i32] {
  trait.assoc_type @Item<[!T, !U]> = !T
  func.func @get(%self: i32) -> i64 {
    %c = arith.extsi %self : i32 to i64
    return %c : i64
  }
}
