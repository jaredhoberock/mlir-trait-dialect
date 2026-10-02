// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @g derives @Wrapped[!T] from its @Mark parameter, projects the requirement
// back off it -- the evidence @Wrapped_any returns, which is that parameter --
// and selects the parameter over the projection. The call supplies @Mark[i32]
// by one proof, @Nine; the projection reads @Nine back through the derive's
// premise, so the select's arms agree and the call runs @Nine's method.
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
func.func private @g(%p: !trait.claim<@Mark[!T]>) -> i64 {
  %c = arith.constant true
  %w = trait.derive @Wrapped[!T] from @Wrapped_any given(%p) : (!trait.claim<@Mark[!T]>)
  %m = trait.project %w[0] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
  %s = arith.select %c, %p, %m : !trait.claim<@Mark[!T]>
  %v = trait.method.call %s @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %p = trait.witness @Nine for @Mark[i32]
  %v = trait.func.call @g(%p) : (!trait.claim<@Mark[i32] by @Nine>) -> i64
  return %v : i64
}
