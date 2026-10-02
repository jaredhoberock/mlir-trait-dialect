// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped states an ordinary requirement @Mark[!T] and a bound requirement
// @Mark at every argument, which @W witnesses by citing @Blanket; @PW
// discharges the ordinary one by @PB, a proof of @Blanket too. @f projects the
// bound requirement at i32. With @Blanket the one impl of @Mark[i32], the
// proof selection holds for the projection is the one the call's evidence
// spells it with, and the instance runs @Blanket.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Blanket for @Mark[!T] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped[!T] where [@Mark[!T], forall [!trait.bound<0>] -> @Mark[!trait.bound<0>]] {}
trait.impl private @W for @Wrapped[i32] witnesses [#trait<witness requirement 1 by @Blanket[!T = !trait.bound<0>]>] {}
trait.proof private @PB proves @Blanket[!T = i32] for @Mark[i32] given []
trait.proof private @PW proves @W[] for @Wrapped[i32] given [@PB, unit]
func.func private @f(%w: !trait.claim<@Wrapped[!T]>) -> i64 {
  %m = trait.project %w[1] for [!T] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
  %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %v = trait.func.call @f(%w) : (!trait.claim<@Wrapped[i32] by @PW>) -> i64
  return %v : i64
}
