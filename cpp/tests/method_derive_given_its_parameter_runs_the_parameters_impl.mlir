// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run derives @Wrapped[i32] from @Wrapped_i32 given @Mark[i32] projected off
// its parameter, whose impl @O returns @Nine (9) for it, while the receiver's
// proof @H supplies its own @Wrapped[i32] through @WR, whose premise is
// @Seven (7). The derive is proven by the proof whose body it is, over the
// evidence its operand carries, so the derived claim's method runs @Nine's
// and the receiver's @WR decides nothing about it.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
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
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) members [@Wrapped.value]
func.func private @Wrapped.value(!trait.claim<@Wrapped[!T]>) -> i64
trait.impl private @Wrapped_i32(%self: !trait.claim<@Wrapped[i32]>, %mark: !trait.claim<@Mark[i32]>) members [@Wrapped.value -> @Wrapped_i32.value] {
}
func.func private @Wrapped_i32.value(%self: !trait.claim<@Wrapped[i32]>, %mark: !trait.claim<@Mark[i32]>) -> i64 {
  %v = trait.func.call @Mark.value(%mark) : (!trait.claim<@Mark[i32]>) -> i64
  return %v : i64
}
trait.proof private @WR {
  %p0 = trait.witness @Seven for @Mark[i32]
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given(%p0) : (!trait.claim<@Mark[i32] by @Seven>)
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.trait private @Outer(!trait.claim<@Outer[!T]>) -> !trait.claim<@Mark[!T]>
trait.impl private @O(%self: !trait.claim<@Outer[i32]>) {
  %nine = trait.witness @Nine for @Mark[i32]
  trait.return %nine : !trait.claim<@Mark[i32] by @Nine>
}
trait.proof private @OP {
  %d = trait.derive @Outer[i32] from @O given()
  trait.return %d : !trait.claim<@Outer[i32]>
}
trait.trait private @Host(!trait.claim<@Host[!T]>) members [@Host.run]
func.func private @Host.run(!trait.claim<@Host[!T]>, !trait.claim<@Outer[!T]>) -> i64
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>, %wrapped: !trait.claim<@Wrapped[i32]>) members [@Host.run -> @Host_i32.run] {
}
func.func private @Host_i32.run(%self: !trait.claim<@Host[i32]>, %wrapped: !trait.claim<@Wrapped[i32]>, %c: !trait.claim<@Outer[i32]>) -> i64 {
  %m = trait.project %c[0] : !trait.claim<@Outer[i32]> -> !trait.claim<@Mark[i32]>
  %w = trait.derive @Wrapped[i32] from @Wrapped_i32 given(%m) : (!trait.claim<@Mark[i32]>)
  %v = trait.func.call @Wrapped.value(%w) : (!trait.claim<@Wrapped[i32]>) -> i64
  return %v : i64
}
trait.proof private @H {
  %p0 = trait.witness @WR for @Wrapped[i32]
  %d = trait.derive @Host[i32] from @Host_i32 given(%p0) : (!trait.claim<@Wrapped[i32] by @WR>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %c = trait.witness @OP for @Outer[i32]
  %v = trait.func.call @Host.run(%h, %c) : (!trait.claim<@Host[i32] by @H>, !trait.claim<@Outer[i32] by @OP>) -> i64
  return %v : i64
}
