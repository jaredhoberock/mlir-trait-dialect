// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @PN and @Alias are two names of one proof: @Nine at i32, given nothing. @f
// calls @Wrapped's evidence method for its quantified requirement at i32, and
// @W's method derives @Mark at any argument from @Nine; the call supplies
// @Mark[i32] by @Alias to another parameter. Nothing respells the method's
// result by the claim it states, so the evidence the method computes stands,
// and the call through it runs @Nine.

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
trait.proof private @PN {
  %d = trait.derive @Mark[i32] from @Nine[i32] given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.proof private @Alias {
  %d = trait.derive @Mark[i32] from @Nine[i32] given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) members [@Wrapped.requirement_0]
func.func private @Wrapped.requirement_0(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!B]>
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) members [@Wrapped.requirement_0 -> @W.requirement_0] {
}
func.func private @W.requirement_0(%self: !trait.claim<@Wrapped[i32]>) -> !trait.claim<@Mark[!trait.poly<0>]> {
  %r = trait.derive @Mark[!trait.poly<0>] from @Nine[!trait.poly<0>] given()
  return %r : !trait.claim<@Mark[!trait.poly<0>]>
}
trait.proof private @PW {
  %d = trait.derive @Wrapped[i32] from @W given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%w: !trait.claim<@Wrapped[!T]>, %p: !trait.claim<@Mark[!T]>) -> i64 {
  %m = trait.func.call @Wrapped.requirement_0(%w) : (!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[i32]>
  %v = trait.func.call @Mark.value(%m) : (!trait.claim<@Mark[i32]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %p = trait.witness @Alias for @Mark[i32]
  %v = trait.func.call @f(%w, %p) : (!trait.claim<@Wrapped[i32] by @PW>, !trait.claim<@Mark[i32] by @Alias>) -> i64
  return %v : i64
}
