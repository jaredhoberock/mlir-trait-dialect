// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Host's default @run, copied into @Host_i32, derives @Wrapped from its @Mark
// parameter, projects the requirement back off it, and selects between the
// two. The call supplies @Mark[i32] by one proof, @Nine, so the instance
// spells the projection with it while the derive is still to be proven, and
// the stage witnesses the projection only once the derive supplies the same
// proof. The select's arms agree and the call runs @Nine's method.
// The condition is a constant: claim erasure has no rule for a select of two
// distinct claim values, so only a select the folder removes reaches it.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> {}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Wrapped_any(%self: !trait.claim<@Wrapped[!T]>, %mark: !trait.claim<@Mark[!T]>) {
  trait.return %mark : !trait.claim<@Mark[!T]>
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) {
  trait.method @run(%p: !trait.claim<@Mark[!T]>) -> i64 {
    %c = arith.constant false
    %w = trait.derive @Wrapped[!T] from @Wrapped_any[!trait.poly<0>] given(%p) : (!trait.claim<@Mark[!T]>)
    %m = trait.project %w[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
    %s = arith.select %c, %p, %m : !trait.claim<@Mark[!T]>
    %v = trait.method.call %s @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>) {}
func.func @main() -> i64 {
  %h = trait.witness @Host_i32 for @Host[i32]
  %p = trait.witness @Nine for @Mark[i32]
  %v = trait.method.call %h @Host[i32]::@run(%p) : (!trait.claim<@Mark[i32] by @Nine>) -> i64 by @Host_i32
  return %v : i64
}
