// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run alleges @Outer[i32], projects @Mark[i32] off it inside an
// scf.execute_region and derives @Wrapped[i32] from the region's result. With
// @Nine the one impl of @Mark[i32], the projection the derive waits for finds
// the proof the instance spells it with, and the instance runs @Nine.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) { trait.method @value() -> i64 }
trait.impl private @Wrapped_i32(%self: !trait.claim<@Wrapped[i32]>, %mark: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = trait.method.call %mark @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @WR {
  %p0 = trait.witness @Nine for @Mark[i32]
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given(%p0) : (!trait.claim<@Mark[i32] by @Nine>)
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.trait private @Outer(%self: !trait.claim<@Outer[!T]>) -> !trait.claim<@Mark[!T]> {}
trait.impl private @O(%self: !trait.claim<@Outer[i32]>) {
  %req0 = trait.allege @Mark[i32]
  trait.return %req0 : !trait.claim<@Mark[i32]>
}
trait.proof private @OP {
  %d = trait.derive @Outer[i32] from @O given()
  trait.return %d : !trait.claim<@Outer[i32]>
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) { trait.method @run() -> i64 }
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>, %wrapped: !trait.claim<@Wrapped[i32]>) {
  trait.method @run() -> i64 {
    %k = scf.execute_region -> !trait.claim<@Mark[i32]> {
      %a = trait.allege @Outer[i32]
      %m = trait.project %a[0] : !trait.claim<@Outer[i32]> -> !trait.claim<@Mark[i32]>
      scf.yield %m : !trait.claim<@Mark[i32]>
    }
    %w = trait.derive @Wrapped[i32] from @Wrapped_i32 given(%k) : (!trait.claim<@Mark[i32]>)
    %v = trait.method.call %w @Wrapped[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H {
  %p0 = trait.witness @WR for @Wrapped[i32]
  %d = trait.derive @Host[i32] from @Host_i32 given(%p0) : (!trait.claim<@Wrapped[i32] by @WR>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %v = trait.method.call %h @Host[i32]::@run() : () -> i64 by @H
  return %v : i64
}
