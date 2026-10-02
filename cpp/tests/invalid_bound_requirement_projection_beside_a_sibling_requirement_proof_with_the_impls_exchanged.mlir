// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @Wrapped states an ordinary requirement @Mark[!T] and a bound requirement
// @Mark at every argument, which @W witnesses by citing @Blanket; @PW
// discharges the ordinary one by @Specific. @f projects the bound requirement
// at i32, and the call's evidence holds one proof of @Mark[i32], @Specific,
// found at the ordinary requirement, so the instance spells the projection
// with it. No subproof names the bound requirement's proof: its evidence is
// the proof selection holds for @Mark[i32], and selection meets two impls of
// it. The projection is refused rather than run through the proof of another
// requirement, whichever of the two impls returns which value.

// CHECK: error: incoherent impls (multiple satisfiable) for '!trait.claim<@Mark[i32]>'

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Specific for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Blanket for @Mark[!T] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.trait private @Wrapped[!T] where [@Mark[!T], forall [!trait.bound<0>] -> @Mark[!trait.bound<0>]] {}
trait.impl private @W for @Wrapped[i32] witnesses [#trait<witness requirement 1 by @Blanket[!T = !trait.bound<0>]>] {}
trait.proof private @PW proves @W[] for @Wrapped[i32] given [@Specific, unit]
func.func private @f(%w: !trait.claim<@Wrapped[!T]>) -> i64 {
  %m = trait.project %w[1] for [!T] : !trait.claim<@Wrapped[!T]> -> !trait.claim<@Mark[!T]>
  %v = trait.method.call %m @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %w = trait.witness @PW for @Wrapped[i32]
  %v = trait.func.call @f(%w) : (!trait.claim<@Wrapped[i32] by @PW>) -> i64
  return %v : i64
}
