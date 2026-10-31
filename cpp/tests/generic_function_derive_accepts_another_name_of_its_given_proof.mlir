// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @PF and @PF2 are two names of one proof: @MT at i32, given @Nine. @f
// derives @Wrapped from @W given its parameter, which the call supplies by
// @PF2, while the proof selection holds for @Wrapped[tuple<i32>] discharges
// @W's entry by @PF. The two names are one piece of evidence, so the derive
// keeps its commitment and runs @MT over @Nine.

// CHECK: {{^}}10{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Nine.value] {
}
func.func private @Nine.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.impl private @MT(%self: !trait.claim<@Mark[tuple<!T>]>, %mark: !trait.claim<@Mark[!T]>) members [@Mark.value -> @MT.value] {
}
func.func private @MT.value(%self: !trait.claim<@Mark[tuple<!T>]>, %mark: !trait.claim<@Mark[!T]>) -> i64 {
  %v = trait.func.call @Mark.value(%mark) : (!trait.claim<@Mark[!T]>) -> i64
  %one = arith.constant 1 : i64
  %r = arith.addi %v, %one : i64
  return %r : i64
}
trait.trait private @Wrapped(!trait.claim<@Wrapped[!T]>) members [@Wrapped.value]
func.func private @Wrapped.value(!trait.claim<@Wrapped[!T]>) -> i64
trait.impl private @W(%self: !trait.claim<@Wrapped[!T]>, %mark: !trait.claim<@Mark[!T]>) members [@Wrapped.value -> @W.value] {
}
func.func private @W.value(%self: !trait.claim<@Wrapped[!T]>, %mark: !trait.claim<@Mark[!T]>) -> i64 {
  %v = trait.func.call @Mark.value(%mark) : (!trait.claim<@Mark[!T]>) -> i64
  return %v : i64
}
trait.proof private @PF {
  %p0 = trait.witness @Nine for @Mark[i32]
  %d = trait.derive @Mark[tuple<i32>] from @MT[i32] given(%p0) : (!trait.claim<@Mark[i32] by @Nine>)
  trait.return %d : !trait.claim<@Mark[tuple<i32>]>
}
trait.proof private @PF2 {
  %p0 = trait.witness @Nine for @Mark[i32]
  %d = trait.derive @Mark[tuple<i32>] from @MT[i32] given(%p0) : (!trait.claim<@Mark[i32] by @Nine>)
  trait.return %d : !trait.claim<@Mark[tuple<i32>]>
}
func.func private @f(%p: !trait.claim<@Mark[!T]>) -> i64 {
  %w = trait.derive @Wrapped[!T] from @W[!trait.poly<0>] given(%p) : (!trait.claim<@Mark[!T]>)
  %v = trait.func.call @Wrapped.value(%w) : (!trait.claim<@Wrapped[!T]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %q = trait.witness @PF2 for @Mark[tuple<i32>]
  %y = trait.func.call @f(%q) : (!trait.claim<@Mark[tuple<i32>] by @PF2>) -> i64
  return %y : i64
}
