// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Trait requires itself at its associated type, which @I binds to i32, so a
// proof of @I discharges that requirement by a proof of @I again: @P1 cites
// itself, and @P2 is the same proof under another name. @main derives @Other
// from @W given @P2, while @PW discharges @W's entry by @P1. Compared entry by
// entry the two names never differ, so they are one piece of evidence, the
// derive keeps its commitment, and the comparison ends at the cycle.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Trait[!T] where [@Trait[!trait.proj<@Trait[!T], "Sub">]] {
  trait.assoc_type @Sub
}
trait.impl private @I for @Trait[i32] {
  trait.assoc_type @Sub = i32
}
trait.proof private @P1 proves @I[] for @Trait[i32] given [@P1]
trait.proof private @P2 proves @I[] for @Trait[i32] given [@P2]
trait.trait private @Other[!T] { trait.method @method() -> i64 }
trait.impl private @W for @Other[!T] where [@Trait[!T]] {
  trait.method @method() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.proof private @PW proves @W[!T = i32] for @Other[i32] given [@P1]
func.func @main() -> i64 {
  %p = trait.witness @P2 for @Trait[i32]
  %d = trait.derive @Other[i32] from @W[!T = i32] given(%p) : (!trait.claim<@Trait[i32] by @P2>)
  %v = trait.method.call %d @Other[i32]::@method() : () -> i64
  return %v : i64
}
