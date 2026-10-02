// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @run alleges @Wrapped[i32], which selection proves by @W9 (premise @Nine),
// and projects @Mark[i32] off the allegation. The call supplies @Mark[i32] by
// the receiver's proof alone, @H discharging the impl's own entry with @Seven,
// so the instance spells the projection with @Seven while its source is still
// an allegation. Once the allegation is proven, its source cites @Nine at the
// index: the spelled proof is not the one the source supplies, and the
// projection is refused rather than run through @Seven.

// CHECK: error: 'trait.project' op names @Seven, which its source does not supply at index 0: the evidence there is @Nine

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.trait private @Wrapped[!T] where [@Mark[!T]] {}
trait.impl private @Seven for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Wrapped_i32 for @Wrapped[i32] {}
trait.proof private @W9 proves @Wrapped_i32[] for @Wrapped[i32] given [@Nine]
trait.trait private @Host[!T] { trait.method @run() -> i64 }
trait.impl private @Host_i32 for @Host[i32] where [@Mark[i32]] {
  trait.method @run() -> i64 {
    %a = trait.allege @Wrapped[i32]
    %m = trait.project %a[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %v = trait.method.call %h @Host[i32]::@run() : () -> i64 by @H
  return %v : i64
}
