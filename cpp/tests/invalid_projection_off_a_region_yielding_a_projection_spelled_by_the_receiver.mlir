// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @run alleges @Outer[i32], which selection proves by @OP (premise @W9,
// whose premise is @Nine), projects @Wrapped[i32] off it inside an
// scf.execute_region, and projects @Mark[i32] off the region's result. The
// call supplies @Wrapped[i32] by the receiver's @W7 alone, whose premise is
// @Seven, so the instance spells the inner projection and the region's result
// with @W7 and the outer projection with @Seven, which agree with each other.
// The outer projection waits for the projection the region yields, and that
// one, once its allegation is proven, finds @W9 at its index: it is refused
// rather than confirming the outer projection and being erased with the
// region.

// CHECK: error: 'trait.project' op names @W7, which its source does not supply at index 0: the evidence there is @W9

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
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
trait.trait private @Wrapped[!T] where [@Mark[!T]] {}
trait.impl private @Wrapped_i32 for @Wrapped[i32] {}
trait.proof private @W9 proves @Wrapped_i32[] for @Wrapped[i32] given [@Nine]
trait.proof private @W7 proves @Wrapped_i32[] for @Wrapped[i32] given [@Seven]
trait.trait private @Outer[!T] where [@Wrapped[!T]] {}
trait.impl private @O for @Outer[i32] {}
trait.proof private @OP proves @O[] for @Outer[i32] given [@W9]
trait.trait private @Host[!T] { trait.method @run() -> i64 }
trait.impl private @Host_i32 for @Host[i32] where [@Wrapped[i32]] {
  trait.method @run() -> i64 {
    %k = scf.execute_region -> !trait.claim<@Wrapped[i32]> {
      %a = trait.allege @Outer[i32]
      %w = trait.project %a[0] : !trait.claim<@Outer[i32]> -> !trait.claim<@Wrapped[i32]>
      scf.yield %w : !trait.claim<@Wrapped[i32]>
    }
    %m = trait.project %k[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@W7]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %v = trait.method.call %h @Host[i32]::@run() : () -> i64 by @H
  return %v : i64
}
