// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One body projects @Mark[i32] twice: off its receiver, whose proof @H derives
// @Host_i32, which returns @Seven (7) for @Host's requirement, and off its
// parameter, whose proof @W derives @Wrapped_i32, which returns @Nine (9) for
// @Wrapped's. Each projection reads the evidence its own source's impl returns
// at its index, so the two calls run different methods and the body answers
// 7 * 10 + 9.

// CHECK: {{^}}79{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]>
trait.trait private @Host(!trait.claim<@Host[!T]>) -> !trait.claim<@Mark[!T]> members [@Host.run]
func.func private @Host.run(!trait.claim<@Host[!T]>, !trait.claim<@Wrapped[!T]>) -> i64
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Seven.value] {
}
func.func private @Seven.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Nine.value] {
}
func.func private @Nine.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.impl private @Wrapped_i32(%self: !trait.claim<@Wrapped[i32]>) {
  %nine = trait.witness @Nine for @Mark[i32]
  trait.return %nine : !trait.claim<@Mark[i32] by @Nine>
}
trait.proof private @W {
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>) members [@Host.run -> @Host_i32.run] {
  %seven = trait.witness @Seven for @Mark[i32]
  trait.return %seven : !trait.claim<@Mark[i32] by @Seven>
}
func.func private @Host_i32.run(%self: !trait.claim<@Host[i32]>, %w: !trait.claim<@Wrapped[i32]>) -> i64 {
  %r = trait.project %self[0] : !trait.claim<@Host[i32]> -> !trait.claim<@Mark[i32]>
  %m = trait.project %w[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
  %a = trait.func.call @Mark.value(%r) : (!trait.claim<@Mark[i32]>) -> i64
  %b = trait.func.call @Mark.value(%m) : (!trait.claim<@Mark[i32]>) -> i64
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
trait.proof private @H {
  %d = trait.derive @Host[i32] from @Host_i32 given()
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.func.call @Host.run(%h, %w) : (!trait.claim<@Host[i32] by @H>, !trait.claim<@Wrapped[i32] by @W>) -> i64
  return %v : i64
}
