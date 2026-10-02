// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One body projects @Mark[i32] twice: off its receiver, whose proof @H
// discharges @Host's requirement with @Seven (7), and off its parameter, whose
// proof @W discharges @Wrapped's with @Nine (9). Each projection reads the
// subproof its own source names at its index, so the two calls run different
// methods and the body answers 7 * 10 + 9.

// CHECK: {{^}}79{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.trait private @Wrapped[!T] where [@Mark[!T]] {}
trait.trait private @Host[!T] where [@Mark[!T]] {
  trait.method @run(!trait.claim<@Wrapped[!T]>) -> i64
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
trait.impl private @Host_i32 for @Host[i32] {
  trait.method @run(%w: !trait.claim<@Wrapped[i32]>) -> i64 {
    %s = trait.assume self : !trait.claim<@Host[i32]>
    %r = trait.project %s[0] : !trait.claim<@Host[i32]> -> !trait.claim<@Mark[i32]>
    %m = trait.project %w[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
    %a = trait.method.call %r @Mark[i32]::@value() : () -> i64
    %b = trait.method.call %m @Mark[i32]::@value() : () -> i64
    %ten = arith.constant 10 : i64
    %tens = arith.muli %a, %ten : i64
    %sum = arith.addi %tens, %b : i64
    trait.return %sum : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @W>) -> i64 by @H
  return %v : i64
}
