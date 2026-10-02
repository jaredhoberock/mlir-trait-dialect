// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run alleges @Outer[i32], projects @Mark[i32] off it inside an
// scf.execute_region and derives @Wrapped[i32] from the region's result. With
// @Nine the one impl of @Mark[i32], the projection the derive waits for finds
// the proof the instance spells it with, and the instance runs @Nine.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
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
trait.proof private @WR proves @Wrapped_i32[] for @Wrapped[i32] given [@Nine]
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
