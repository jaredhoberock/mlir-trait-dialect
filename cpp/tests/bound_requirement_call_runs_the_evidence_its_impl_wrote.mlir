// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped requires @Mark[!T] as a result and @Mark at every argument through
// its evidence member @requirement_1. @W returns @Specific's witness for the
// result and derives @Blanket in its evidence member. @f calls the evidence
// member at i32: the call is replaced by @W's body, whose derive is the proof
// it states, so the call runs @Blanket's member, 9, although @Specific also
// implements @Mark[i32] and serves the other requirement.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.impl private @Specific(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Specific.value] {
}
func.func private @Specific.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Blanket(%self: !trait.claim<@Mark[!T]>) members [@Mark.value -> @Blanket.value] {
}
func.func private @Blanket.value(%self: !trait.claim<@Mark[!T]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> members [@Wrapped.requirement_1]
func.func private @Wrapped.requirement_1(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!U]>
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) members [@Wrapped.requirement_1 -> @W.requirement_1] {
  %mark = trait.witness @Specific for @Mark[i32]
  trait.return %mark : !trait.claim<@Mark[i32] by @Specific>
}
func.func private @W.requirement_1(%self: !trait.claim<@Wrapped[i32]>) -> !trait.claim<@Mark[!trait.poly<0>]> {
  %m = trait.derive @Mark[!trait.poly<0>] from @Blanket[!trait.poly<0>] given()
  return %m : !trait.claim<@Mark[!trait.poly<0>]>
}
trait.proof private @PW {
  %d = trait.derive @Wrapped[i32] from @W given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%w: !trait.claim<@Wrapped[!T]>) -> i64 {
  %m = trait.func.call @Wrapped.requirement_1(%w) : (!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]>
  %v = trait.func.call @Mark.value(%m) : (!trait.claim<@Mark[!T]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %v = trait.func.call @f(%w) : (!trait.claim<@Wrapped[i32] by @PW>) -> i64
  return %v : i64
}
