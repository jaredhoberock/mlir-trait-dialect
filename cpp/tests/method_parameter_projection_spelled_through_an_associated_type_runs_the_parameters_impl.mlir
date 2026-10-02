// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped spells its second requirement @Mark over @Has's associated type,
// which @Has_i32 binds to i32. @run projects it off its parameter, whose proof
// @W discharges it by @Nine, while the receiver's proof @H discharges the
// impl's own @Mark[i32] by @Seven. The instance spells the claim by the
// normalized @Mark[i32] that both proofs prove, so it binds neither, and the
// projection takes the subproof its source cites at its index at the
// application the instance spells: the call runs @Nine, and nothing is
// reported along the way.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
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
trait.trait private @Has[!T] { trait.assoc_type @Out }
trait.impl private @Has_i32 for @Has[i32] { trait.assoc_type @Out = i32 }
trait.trait private @Wrapped[!T] where [@Has[!T], @Mark[!trait.proj<@Has[!T], "Out">]] {}
trait.impl private @Wrapped_i32 for @Wrapped[i32] {}
trait.proof private @W proves @Wrapped_i32[] for @Wrapped[i32] given [@Has_i32, @Nine]
trait.trait private @Host[!T] { trait.method @run(!trait.claim<@Wrapped[!T]>) -> i64 }
trait.impl private @Host_i32 for @Host[i32] where [@Mark[i32]] {
  trait.method @run(%w: !trait.claim<@Wrapped[i32]>) -> i64 {
    %m = trait.project %w[1] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[!trait.proj<@Has[i32], "Out">]>
    %v = trait.method.call %m @Mark[!trait.proj<@Has[i32], "Out">]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H proves @Host_i32[] for @Host[i32] given [@Seven]
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @W>) -> i64 by @H
  return %v : i64
}
