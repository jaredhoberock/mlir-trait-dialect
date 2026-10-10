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
trait.trait private @Trait(%self: !trait.claim<@Trait[!T]>) -> !trait.claim<@Trait[!trait.proj<@Trait[!T], "Sub">]> {
  trait.assoc_type @Sub
}
trait.impl private @I(%self: !trait.claim<@Trait[i32]>) {
  trait.assoc_type @Sub = i32
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
trait.trait private @Other(%self: !trait.claim<@Other[!T]>) { trait.method @method() -> i64 }
trait.impl private @W(%self: !trait.claim<@Other[!T]>, %trait: !trait.claim<@Trait[!T]>) {
  trait.method @method() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.proof private @PW {
  %p0 = trait.witness @P1 for @Trait[i32]
  %d = trait.derive @Other[i32] from @W[i32] given(%p0) : (!trait.claim<@Trait[i32] by @P1>)
  trait.return %d : !trait.claim<@Other[i32]>
}
func.func @main() -> i64 {
  %p = trait.witness @P2 for @Trait[i32]
  %d = trait.derive @Other[i32] from @W[i32] given(%p) : (!trait.claim<@Trait[i32] by @P2>)
  %v = trait.method.call %d @Other[i32]::@method() : () -> i64
  return %v : i64
}
