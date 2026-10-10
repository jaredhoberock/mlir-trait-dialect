// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped states an ordinary requirement @Mark[!T] and a quantified
// requirement @Mark at every argument. @W returns @PB, a proof of @Blanket, for
// the ordinary one, and its evidence method derives the quantified one from
// @Blanket. @f calls the evidence method at its own argument; with @Blanket
// the one impl of @Mark[i32], the instance runs @Blanket.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
!B = !trait.poly<1>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Blanket(%self: !trait.claim<@Mark[!T]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> {
  trait.method @requirement_1() -> !trait.claim<@Mark[!B]>
}
trait.proof private @PB {
  %d = trait.derive @Mark[i32] from @Blanket given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  trait.method @requirement_1() -> !trait.claim<@Mark[!trait.poly<0>]> {
    %r = trait.derive @Mark[!trait.poly<0>] from @Blanket given()
    trait.return %r : !trait.claim<@Mark[!trait.poly<0>]>
  }
  %mark = trait.witness @PB for @Mark[i32]
  trait.return %mark : !trait.claim<@Mark[i32] by @PB>
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
