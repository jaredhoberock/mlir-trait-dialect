// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Host's default @run projects @Mark[i32] off its parameter, proven by @W
// through @Wrapped_i32, which returns @Nine (9) for that requirement, and
// @Host_i32 takes the default, its receiver's proof @H discharging the impl's
// own @Mark[i32] entry with @Seven (7). The default cut for the impl reads the
// projection off the parameter's evidence, so the call through it runs
// @Nine's method.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]>
trait.trait private @Host(!trait.claim<@Host[!T]>) members [default @Host.run]
func.func private @Host.run(%self: !trait.claim<@Host[!T]>, %w: !trait.claim<@Wrapped[!T]>) -> i64 {
  %m = trait.project %w[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
  %v = trait.func.call @Mark.value(%m) : (!trait.claim<@Mark[!T]>) -> i64
  return %v : i64
}
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
  %mark = trait.witness @Nine for @Mark[i32]
  trait.return %mark : !trait.claim<@Mark[i32] by @Nine>
}
trait.proof private @W {
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>, %mark: !trait.claim<@Mark[i32]>) members [@Host.run -> @Host.run] {
}
trait.proof private @H {
  %p0 = trait.witness @Seven for @Mark[i32]
  %d = trait.derive @Host[i32] from @Host_i32 given(%p0) : (!trait.claim<@Mark[i32] by @Seven>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.func.call @Host.run(%h, %w) : (!trait.claim<@Host[i32] by @H>, !trait.claim<@Wrapped[i32] by @W>) -> i64
  return %v : i64
}
