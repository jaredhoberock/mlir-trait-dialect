// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run projects @Mark[i32] off its parameter, whose impl @Wrapped_i32 returns
// @Nine's witness for it, and passes the projection through a select of itself,
// while the receiver's proof @H discharges the impl's own @Mark[i32] entry with
// @Seven. The select hands back the value its arms supply, so the call through
// it runs @Nine's member, whatever the receiver supplies for the same claim.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]>
trait.trait private @Host(!trait.claim<@Host[!T]>) members [@Host.run]
func.func private @Host.run(!trait.claim<@Host[!T]>, !trait.claim<@Wrapped[!T]>, i1) -> i64
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
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>, %mark: !trait.claim<@Mark[i32]>) members [@Host.run -> @Host_i32.run] {
}
func.func private @Host_i32.run(%self: !trait.claim<@Host[i32]>, %mark: !trait.claim<@Mark[i32]>, %w: !trait.claim<@Wrapped[i32]>, %c: i1) -> i64 {
  %m = trait.project %w[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
  %s = arith.select %c, %m, %m : !trait.claim<@Mark[i32]>
  %v = trait.func.call @Mark.value(%s) : (!trait.claim<@Mark[i32]>) -> i64
  return %v : i64
}
trait.proof private @H {
  %p0 = trait.witness @Seven for @Mark[i32]
  %d = trait.derive @Host[i32] from @Host_i32 given(%p0) : (!trait.claim<@Mark[i32] by @Seven>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %c = arith.constant false
  %v = trait.func.call @Host.run(%h, %w, %c) : (!trait.claim<@Host[i32] by @H>, !trait.claim<@Wrapped[i32] by @W>, i1) -> i64
  return %v : i64
}
