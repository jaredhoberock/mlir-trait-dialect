// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Host's default @run projects @Mark[i32] off its parameter, whose proof @W
// discharges it with @Nine (9), and @Host_i32 takes the default, its receiver's
// proof @H discharging the impl's own @Mark[i32] entry with @Seven (7). The
// default copied into the impl is cut like a method written there: the
// projection reads @W's subproof, so the call through it runs @Nine's method.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.trait private @Wrapped[!T] where [@Mark[!T]] {}
trait.trait private @Host[!T] {
  trait.method @run(%w: !trait.claim<@Wrapped[!T]>) -> i64 {
    %m = trait.project %w[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
    %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @Seven for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Wrapped_i32 for @Wrapped[i32] {}
trait.proof private @W proves @Wrapped_i32[] for @Wrapped[i32] given [@Nine]
trait.impl private @Host_i32 for @Host[i32] where [@Mark[i32]] {}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @W>) -> i64 by @H
  return %v : i64
}
