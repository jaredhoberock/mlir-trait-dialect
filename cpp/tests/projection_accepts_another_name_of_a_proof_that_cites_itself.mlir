// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Trait requires itself at its associated type, which @I binds to i32, so
// @P1 cites itself, and @P2 is the same proof under another name. @f alleges
// @Other, which selection proves by @PW (premise @P1), and projects @Trait off
// it; the call supplies @Trait by @P2 alone, so the instance spells the
// projection with @P2. Compared entry by entry the two names never differ, so
// the projection names the evidence its source supplies, and the comparison
// ends at the cycle.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Trait[!T] where [@Trait[!trait.proj<@Trait[!T], "Sub">]] {
  trait.assoc_type @Sub
  trait.method @method() -> i64
}
trait.impl private @I for @Trait[i32] {
  trait.assoc_type @Sub = i32
  trait.method @method() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.proof private @P1 proves @I[] for @Trait[i32] given [@P1]
trait.proof private @P2 proves @I[] for @Trait[i32] given [@P2]
trait.trait private @Other[!T] where [@Trait[!T]] {}
trait.impl private @W for @Other[i32] {}
trait.proof private @PW proves @W[] for @Other[i32] given [@P1]
func.func private @f(%p: !trait.claim<@Trait[!T]>) -> i64 {
  %w = trait.allege @Other[!T]
  %d = trait.project %w[0] : !trait.claim<@Other[!T]> -> !trait.claim<@Trait[!T]>
  %v = trait.method.call %d @Trait[!T]::@method() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %p = trait.witness @P2 for @Trait[i32]
  %v = trait.func.call @f(%p) : (!trait.claim<@Trait[i32] by @P2>) -> i64
  return %v : i64
}
