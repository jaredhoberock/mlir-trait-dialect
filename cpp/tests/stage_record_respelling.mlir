// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: env TRAIT_DEMAND_CENSUS=1 mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// Proving a claim leaves every other spelling of it stale, so the stage sweeps
// the module and respells them from what it has recorded. The sweep reports how
// much of the module it touched: how many bindings it applied, how many ops
// carried a spelling it moved, and how many positions on those ops moved.
//
// Here @Choose_i32's method takes the @Convert claim as a parameter, so proving
// @Convert[i32, i32] moves that method's block-argument type and the function
// type beside it -- two positions on the one op that carries them.

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @Convert[!T, !U] {
  func.func nested @convert(!U) -> !T
}

trait.impl @Convert_i32 for @Convert[i32, i32] {
  func.func nested @convert(%x: i32) -> i32 {
    return %x : i32
  }
}

trait.trait @Choose[!T] {
  func.func nested @choose(!T, !trait.claim<@Convert[!T, !T]>) -> !T
}

trait.impl @Choose_i32 for @Choose[i32] {
  func.func nested @choose(%a: i32, %same: !trait.claim<@Convert[i32, i32]>) -> i32 {
    %converted = trait.method.call %same @Convert[i32, i32]::@convert(%a)
      : (i32) -> i32
    return %converted : i32
  }
}

func.func @test(%x: i32) -> i32 {
  %chooser = trait.allege @Choose[i32]
  %same = trait.allege @Convert[i32, i32]
  %res = trait.method.call %chooser @Choose[i32]::@choose(%x, %same)
    : (i32, !trait.claim<@Convert[i32, i32]>) -> i32
  return %res : i32
}

// CHECK: trait-stage-record respelling round=0 bindings=2 ops=1 positions=2
