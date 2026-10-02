// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Wrapped's bound requirement has an equality premise, which @f discharges
// with a reflexive witness rather than a projection. With @Nine the one impl
// of @Mark[i32], selection's proof is the one the call spells the projection
// with, and the instance runs @Nine.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Nine for @Mark[!T] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped[!T] where [!T = !T, forall [!trait.bound<0>] where [!trait.bound<0> = !trait.bound<0>] -> @Mark[!trait.bound<0>]] {}
trait.impl private @W for @Wrapped[!T] witnesses [#trait<witness requirement 1 by @Nine[!T = !trait.bound<0>]>] {}
trait.proof private @PN proves @Nine[!T = i32] for @Mark[i32] given []
trait.proof private @PW proves @W[!T = i32] for @Wrapped[i32] given [unit, unit]
func.func private @f(%w: !trait.claim<@Wrapped[!T]>, %p: !trait.claim<@Mark[!T]>) -> i64 {
  %e = trait.witness refl : !trait.claim<!T = !T>
  %m = trait.project %w[1] for [!T] given(%e : !trait.claim<!T = !T>) : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
  %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %p = trait.witness @PN for @Mark[i32]
  %v = trait.func.call @f(%w, %p) : (!trait.claim<@Wrapped[i32] by @PW>, !trait.claim<@Mark[i32] by @PN>) -> i64
  return %v : i64
}
