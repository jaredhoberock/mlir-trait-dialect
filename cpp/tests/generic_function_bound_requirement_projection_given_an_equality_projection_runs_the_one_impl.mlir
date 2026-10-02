// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped's quantified requirement has an equality premise, which @f
// discharges with a projection of @Wrapped's equality requirement. The call
// supplies @Mark[i32] by @PN to another parameter; the evidence method @W
// implements derives @Mark from @Nine, and the instance runs @Nine.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
!B = !trait.poly<1>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Nine(%self: !trait.claim<@Mark[!T]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<!T = !T> {
  trait.method @requirement_1(!trait.claim<!B = !B>) -> !trait.claim<@Mark[!B]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[!T]>) {
  trait.method @requirement_1(%e: !trait.claim<!B = !B>) -> !trait.claim<@Mark[!B]> {
    %r = trait.derive @Mark[!B] from @Nine given()
    trait.return %r : !trait.claim<@Mark[!B]>
  }
  %refl = trait.witness refl : !trait.claim<!T = !T>
  trait.return %refl : !trait.claim<!T = !T>
}
trait.proof private @PN {
  %d = trait.derive @Mark[i32] from @Nine given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.proof private @PW {
  %d = trait.derive @Wrapped[i32] from @W given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%w: !trait.claim<@Wrapped[!T]>, %p: !trait.claim<@Mark[!T]>) -> i64 {
  %e = trait.project %w[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<!T = !T>
  %m = trait.method.call %w @Wrapped[!T]::@requirement_1(%e) : (!trait.claim<!T = !T>) -> !trait.claim<@Mark[!T]>
  %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %p = trait.witness @PN for @Mark[i32]
  %v = trait.func.call @f(%w, %p) : (!trait.claim<@Wrapped[i32] by @PW>, !trait.claim<@Mark[i32] by @PN>) -> i64
  return %v : i64
}
