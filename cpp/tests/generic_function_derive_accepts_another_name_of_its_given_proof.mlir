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
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @MT for @Mark[tuple<!T>] where [@Mark[!T]] {
  trait.method @value() -> i64 {
    %p = trait.assume 0 : !trait.claim<@Mark[!T]>
    %v = trait.method.call %p @Mark[!T]::@value() : () -> i64
    %one = arith.constant 1 : i64
    %r = arith.addi %v, %one : i64
    trait.return %r : i64
  }
}
trait.trait private @Wrapped[!T] { trait.method @value() -> i64 }
trait.impl private @W for @Wrapped[!T] where [@Mark[!T]] {
  trait.method @value() -> i64 {
    %p = trait.assume 0 : !trait.claim<@Mark[!T]>
    %v = trait.method.call %p @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @PF proves @MT[!T = i32] for @Mark[tuple<i32>] given [@Nine]
trait.proof private @PF2 proves @MT[!T = i32] for @Mark[tuple<i32>] given [@Nine]
func.func private @f(%p: !trait.claim<@Mark[!T]>) -> i64 {
  %w = trait.derive @Wrapped[!T] from @W[!T = !T] given(%p) : (!trait.claim<@Mark[!T]>)
  %v = trait.method.call %w @Wrapped[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %q = trait.witness @PF2 for @Mark[tuple<i32>]
  %y = trait.func.call @f(%q) : (!trait.claim<@Mark[tuple<i32>] by @PF2>) -> i64
  return %y : i64
}
