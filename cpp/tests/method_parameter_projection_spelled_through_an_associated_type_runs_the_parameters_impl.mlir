// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped spells its second requirement @Mark over @Has's associated type,
// which @Has_i32 binds to i32. @Wrapped_i32 returns @Nine for it, coerced to
// the trait's spelling through @Has_i32's binding. @run projects it off its
// parameter, while the receiver's proof @H discharges the impl's own
// @Mark[i32] by @Seven. The projection takes the evidence its source's impl
// returns at its index at the application the instance spells: the call runs
// @Nine, and nothing is reported along the way.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Has(%self: !trait.claim<@Has[!T]>) { trait.assoc_type @Out }
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) { trait.assoc_type @Out = i32 }
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> (!trait.claim<@Has[!T]>, !trait.claim<@Mark[!trait.proj<@Has[!T], "Out">]>) {}
trait.impl private @Wrapped_i32(%self: !trait.claim<@Wrapped[i32]>) {
  %has = trait.witness @Has_i32 for @Has[i32]
  %nine = trait.witness @Nine for @Mark[i32]
  %out = trait.witness proj_resolve !trait.proj<@Has[i32], "Out"> resolves i32 by @Has_i32 : !trait.claim<!trait.proj<@Has[i32], "Out"> = i32>
  %mark = trait.coerce %nine : !trait.claim<@Mark[i32] by @Nine> to !trait.claim<@Mark[!trait.proj<@Has[i32], "Out">]> via (%out) : (!trait.claim<!trait.proj<@Has[i32], "Out"> = i32>)
  trait.return %has, %mark : !trait.claim<@Has[i32] by @Has_i32>, !trait.claim<@Mark[!trait.proj<@Has[i32], "Out">]>
}
trait.proof private @W {
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
trait.trait private @Host(%self: !trait.claim<@Host[!T]>) { trait.method @run(!trait.claim<@Wrapped[!T]>) -> i64 }
trait.impl private @Host_i32(%self: !trait.claim<@Host[i32]>, %mark: !trait.claim<@Mark[i32]>) {
  trait.method @run(%w: !trait.claim<@Wrapped[i32]>) -> i64 {
    %m = trait.project %w[1] : !trait.claim<@Wrapped[i32]> -> !trait.claim<@Mark[!trait.proj<@Has[i32], "Out">]>
    %v = trait.method.call %m @Mark[!trait.proj<@Has[i32], "Out">]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @H {
  %p0 = trait.witness @Seven for @Mark[i32]
  %d = trait.derive @Host[i32] from @Host_i32 given(%p0) : (!trait.claim<@Mark[i32] by @Seven>)
  trait.return %d : !trait.claim<@Host[i32]>
}
func.func @main() -> i64 {
  %h = trait.witness @H for @Host[i32]
  %w = trait.witness @W for @Wrapped[i32]
  %v = trait.method.call %h @Host[i32]::@run(%w) : (!trait.claim<@Wrapped[i32] by @W>) -> i64 by @H
  return %v : i64
}
