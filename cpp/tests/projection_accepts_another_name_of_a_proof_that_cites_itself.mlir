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
trait.trait private @Trait(%self: !trait.claim<@Trait[!T]>) -> !trait.claim<@Trait[!trait.proj<@Trait[!T], "Sub">]> {
  trait.assoc_type @Sub
  trait.method @method() -> i64
}
trait.impl private @I(%self: !trait.claim<@Trait[i32]>) {
  trait.assoc_type @Sub = i32
  trait.method @method() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
  %req0 = trait.allege @Trait[!trait.proj<@Trait[i32], "Sub">]
  trait.return %req0 : !trait.claim<@Trait[!trait.proj<@Trait[i32], "Sub">]>
}
trait.proof private @P1 {
  %d = trait.derive @Trait[i32] from @I given()
  trait.return %d : !trait.claim<@Trait[i32]>
}
trait.proof private @P2 {
  %d = trait.derive @Trait[i32] from @I given()
  trait.return %d : !trait.claim<@Trait[i32]>
}
trait.trait private @Other(%self: !trait.claim<@Other[!T]>) -> !trait.claim<@Trait[!T]> {}
trait.impl private @W(%self: !trait.claim<@Other[i32]>) {
  %req0 = trait.allege @Trait[i32]
  trait.return %req0 : !trait.claim<@Trait[i32]>
}
trait.proof private @PW {
  %d = trait.derive @Other[i32] from @W given()
  trait.return %d : !trait.claim<@Other[i32]>
}
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
