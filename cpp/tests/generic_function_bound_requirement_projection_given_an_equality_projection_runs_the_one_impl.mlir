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
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.impl private @Nine(%self: !trait.claim<@Mark[!T]>) members [@Mark.value -> @Nine.value] {
}
func.func private @Nine.value(%self: !trait.claim<@Mark[!T]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) -> !trait.claim<!T = !T> members [@Wrapped.requirement_1]
func.func private @Wrapped.requirement_1(!trait.claim<@Wrapped[!T]>, !trait.claim<!B = !B>) -> !trait.claim<@Mark[!B]>
trait.impl private @W(%self: !trait.claim<@Wrapped[!T]>) members [@Wrapped.requirement_1 -> @W.requirement_1] {
  %refl = trait.witness refl : !trait.claim<!T = !T>
  trait.return %refl : !trait.claim<!T = !T>
}
func.func private @W.requirement_1(%self: !trait.claim<@Wrapped[!T]>, %e: !trait.claim<!B = !B>) -> !trait.claim<@Mark[!B]> {
  %r = trait.derive @Mark[!B] from @Nine[!trait.poly<1>] given()
  return %r : !trait.claim<@Mark[!B]>
}
trait.proof private @PN {
  %d = trait.derive @Mark[i32] from @Nine[i32] given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.proof private @PW {
  %d = trait.derive @Wrapped[i32] from @W[i32] given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%w: !trait.claim<@Wrapped[!T]>, %p: !trait.claim<@Mark[!T]>) -> i64 {
  %e = trait.project %w[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<!T = !T>
  %m = trait.func.call @Wrapped.requirement_1(%w, %e) : (!trait.claim<@Wrapped[!T]>, !trait.claim<!T = !T>) -> !trait.claim<@Mark[!T]>
  %v = trait.func.call @Mark.value(%m) : (!trait.claim<@Mark[!T]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %p = trait.witness @PN for @Mark[i32]
  %v = trait.func.call @f(%w, %p) : (!trait.claim<@Wrapped[i32] by @PW>, !trait.claim<@Mark[i32] by @PN>) -> i64
  return %v : i64
}
