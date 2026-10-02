// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A call of a quantified requirement's method is replaced by the body its
// receiver's impl defines, so the evidence it computes is the operand that body
// returns. @W's @requirement_0 returns its parameter, given @Nine's witness;
// a call spelling its result's proof as @Seven states a decision the IR does
// not make, and the coercion bridging the returned operand to that spelling,
// located at the body's return, refuses to exchange one proof for the other.

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
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) {
  trait.method @requirement_0(!trait.claim<@Mark[!T]>) -> !trait.claim<@Mark[!T]>
}
trait.impl private @W(%self: !trait.claim<@Wrapped[i32]>) {
  trait.method @requirement_0(%p: !trait.claim<@Mark[i32]>) -> !trait.claim<@Mark[i32]> {
    // expected-error @below {{may not swap the proof backing claim #trait<application@Mark[i32]>: a coerce compares modulo a proof but may not exchange it for another}}
    trait.return %p : !trait.claim<@Mark[i32]>
  }
}
func.func @main() -> i64 {
  %w = trait.witness @W for @Wrapped[i32]
  %p = trait.witness @Nine for @Mark[i32]
  %m = trait.method.call %w @Wrapped[i32]::@requirement_0(%p) : (!trait.claim<@Mark[i32] by @Nine>) -> !trait.claim<@Mark[i32] by @Seven> by @W
  %v = trait.method.call %m @Mark[i32]::@value() : () -> i64 by @Seven
  return %v : i64
}
