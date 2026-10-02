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
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Nine for @Mark[!T] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.proof private @PN proves @Nine[!T = i32] for @Mark[i32] given []
trait.proof private @Alias proves @Nine[!T = i32] for @Mark[i32] given []
trait.trait private @Wrapped[!T] where [@Mark[!T]] { trait.method @value() -> i64 }
trait.impl private @W for @Wrapped[!T] where [@Mark[!T]] {
  trait.method @value() -> i64 {
    %p = trait.assume 0 : !trait.claim<@Mark[!T]>
    %v = trait.method.call %p @Mark[!T]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @PW proves @W[!T = i32] for @Wrapped[i32] given [@PN, @PN]
func.func private @f(%p: !trait.claim<@Mark[!T]>) -> i64 {
  %w = trait.derive @Wrapped[!T] from @W[!T = !T] given(%p) : (!trait.claim<@Mark[!T]>)
  %v = trait.method.call %w @Wrapped[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %p = trait.witness @Alias for @Mark[i32]
  %v = trait.func.call @f(%p) : (!trait.claim<@Mark[i32] by @Alias>) -> i64
  return %v : i64
}
