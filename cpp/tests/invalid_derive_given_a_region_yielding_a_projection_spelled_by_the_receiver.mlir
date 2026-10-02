// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @run alleges @Outer[i32], which selection proves by @OP (premise @Nine),
// projects @Mark[i32] off it inside an scf.execute_region, and derives
// @Wrapped[i32] from the region's result. The call supplies @Mark[i32] by the
// receiver's proof alone, @H discharging its entry with @WR and @WR its own
// with @Seven, so the instance spells the projection, the region's result and
// the derive with them. The region's result is what its yield hands back, so
// the derive waits for the projection inside, and the projection, once its
// allegation is proven, finds @Nine at its index: it is refused rather than
// confirming the derive and being erased with the region.

// CHECK: error: 'trait.project' op names @Seven, which its source does not supply at index 0: the evidence there is @Nine

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
trait.trait private @Wrapped[!T] { trait.method @value() -> i64 }
trait.impl private @Wrapped_i32 for @Wrapped[i32] where [@Mark[i32]] {
  trait.method @value() -> i64 {
    %m = trait.assume 0 : !trait.claim<@Mark[i32]>
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @WR proves @Wrapped_i32[] for @Wrapped[i32] given [@Seven]
trait.trait private @Outer[!T] where [@Mark[!T]] {}
trait.impl private @O for @Outer[i32] {}
trait.proof private @OP proves @O[] for @Outer[i32] given [@Nine]
trait.trait private @Host[!T] { trait.method @run() -> i64 }
trait.impl private @Host_i32 for @Host[i32] where [@Wrapped[i32]] {
  trait.method @run() -> i64 {
    %k = scf.execute_region -> !trait.claim<@Mark[i32]> {
      %a = trait.allege @Outer[i32]
      %m = trait.project %a[0] : !trait.claim<@Outer[i32]> -> !trait.claim<@Mark[i32]>
      scf.yield %m : !trait.claim<@Mark[i32]>
    }
    %w = trait.derive @Wrapped[i32] from @Wrapped_i32[] given(%k) : (!trait.claim<@Mark[i32]>)
    %v = trait.method.call %w @Wrapped[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@WR]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %v = trait.method.call %h @Host[i32]::@run() : () -> i64 by @H
  return %v : i64
}
