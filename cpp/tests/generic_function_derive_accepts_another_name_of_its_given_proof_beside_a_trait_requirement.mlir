// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @PN and @Alias are two names of one proof: @Nine at i32, given nothing.
// @Wrapped requires @Mark[!T] and @W assumes it too, so @PW's subproofs are
// the trait's requirement and then the impl's where-clause entry, both @PN.
// @f derives @Wrapped from @W given its parameter, which the call supplies by
// @Alias: the where-clause entry, read after the trait's requirement, names
// the same evidence, so the derive keeps its commitment and runs @W.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Nine(%self: !trait.claim<@Mark[!T]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.proof private @PN {
  %d = trait.derive @Mark[i32] from @Nine[i32] given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.proof private @Alias {
  %d = trait.derive @Mark[i32] from @Nine[i32] given()
  trait.return %d : !trait.claim<@Mark[i32]>
}
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> { trait.method @value() -> i64 }
trait.impl private @W(%self: !trait.claim<@Wrapped[!T]>, %mark: !trait.claim<@Mark[!T]>) {
  trait.method @value() -> i64 {
    %v = trait.method.call %mark @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
  trait.return %mark : !trait.claim<@Mark[!T]>
}
trait.proof private @PW {
  %p0 = trait.witness @PN for @Mark[i32]
  %d = trait.derive @Wrapped[i32] from @W[i32] given(%p0) : (!trait.claim<@Mark[i32] by @PN>)
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func private @f(%p: !trait.claim<@Mark[!T]>) -> i64 {
  %w = trait.derive @Wrapped[!T] from @W[!trait.poly<0>] given(%p) : (!trait.claim<@Mark[!T]>)
  %v = trait.method.call %w @Wrapped[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %p = trait.witness @Alias for @Mark[i32]
  %v = trait.func.call @f(%p) : (!trait.claim<@Mark[i32] by @Alias>) -> i64
  return %v : i64
}
