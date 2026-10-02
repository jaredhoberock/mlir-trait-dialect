// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @run derives @Wrapped[i32] from its @Mark parameter, projects the
// requirement back off it, and selects between the parameter and the
// projection. The call supplies @Mark[i32] by one proof, @Nine, so the instance
// spells the projection with it while the derive is still to be proven, and
// the stage witnesses the projection only once the derive supplies the same
// proof. The select's arms agree and the call runs @Nine's method.
// The condition is a constant: claim erasure has no rule for a select of two
// distinct claim values, so only a select the folder removes reaches it.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.trait private @Wrapped[!T] where [@Mark[!T]] {}
trait.impl private @Nine for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Wrapped_any for @Wrapped[!T] where [@Mark[!T]] {}
trait.trait private @Host[!T] { trait.method @run(!trait.claim<@Mark[!T]>) -> i64 }
trait.impl private @Host_i32 for @Host[i32] {
  trait.method @run(%p: !trait.claim<@Mark[i32]>) -> i64 {
    %c = arith.constant false
    %w = trait.derive @Wrapped[i32] from @Wrapped_any[!T = i32] given(%p) : (!trait.claim<@Mark[i32]>)
    %m = trait.project %w[0] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[i32]>
    %s = arith.select %c, %p, %m : !trait.claim<@Mark[i32]>
    %v = trait.method.call %s @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
func.func @main() -> i64 {
  %h = trait.witness @Host_i32 for @Host[i32]
  %p = trait.witness @Nine for @Mark[i32]
  %v = trait.method.call %h @Host[i32]::@run(%p) : (!trait.claim<@Mark[i32] by @Nine>) -> i64 by @Host_i32
  return %v : i64
}
