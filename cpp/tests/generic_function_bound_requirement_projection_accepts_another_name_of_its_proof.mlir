// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @PN and @Alias are two names of one proof: @Nine at i32, given nothing. @f
// projects @Wrapped's bound requirement at i32, which no subproof names, so
// its evidence is the proof selection holds for @Mark[i32], @PN; the call
// supplies @Mark[i32] by @Alias alone, so the instance spells the projection
// with @Alias. The two names are one piece of evidence, and the projection
// runs @Nine.

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
trait.trait private @Wrapped[!T] where [forall [!trait.bound<0>] -> @Mark[!trait.bound<0>]] {}
trait.impl private @W for @Wrapped[i32] witnesses [#trait<witness requirement 0 by @Nine[!T = !trait.bound<0>]>] {}
trait.proof private @PW proves @W[] for @Wrapped[i32] given [unit]
func.func private @f(%w: !trait.claim<@Wrapped[!T]>, %p: !trait.claim<@Mark[!T]>) -> i64 {
  %m = trait.project %w[0] for [i32] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[i32]>
  %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %p = trait.witness @Alias for @Mark[i32]
  %v = trait.func.call @f(%w, %p) : (!trait.claim<@Wrapped[i32] by @PW>, !trait.claim<@Mark[i32] by @Alias>) -> i64
  return %v : i64
}
