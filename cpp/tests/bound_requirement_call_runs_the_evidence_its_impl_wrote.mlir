// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped requires @Mark[!T] as a result and @Mark at every argument through
// its evidence method @requirement_1. @W returns @Specific's witness for the
// result and derives @Blanket in its evidence method. @f calls the evidence
// method at i32: the call is replaced by @W's body, whose derive is the proof
// it states, so the call runs @Blanket's method, 9, although @Specific also
// implements @Mark[i32] and serves the other requirement.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Specific(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Blanket(%self: !trait.claim<@Mark[!T]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> {
  trait.method @requirement_1() -> !trait.claim<@Mark[!U]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  %mark = trait.witness @Specific for @Mark[i32]
  trait.method @requirement_1() -> !trait.claim<@Mark[!U]> {
    %m = trait.derive @Mark[!U] from @Blanket given()
    trait.return %m : !trait.claim<@Mark[!U]>
  }
  trait.return %mark : !trait.claim<@Mark[i32] by @Specific>
}
trait.proof private @PW {
  %d = trait.derive @Wrapped[i32] from @W given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%w: !trait.claim<@Wrapped[!T]>) -> i64 {
  %m = trait.method.call %w @Wrapped[!T]::@requirement_1() : () -> !trait.claim<@Mark[!T]>
  %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %v = trait.func.call @f(%w) : (!trait.claim<@Wrapped[i32] by @PW>) -> i64
  return %v : i64
}
