// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @Wrapped_i32 returns @Nine's witness for its trait's requirement @Mark[i32],
// so a claim of @Wrapped[i32] proven by @W9 supplies @Mark[i32] by @Nine at
// index 0. A projection off it spelled with @Seven names a proof its source
// does not supply: the stage replaces the projection by @Nine's witness,
// coerced to the projection's spelling, and the coercion refuses to exchange
// one proof for the other.

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.trait private @Wrapped(%self: !trait.claim<@Wrapped[!T]>) -> !trait.claim<@Mark[!T]> {}
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
trait.impl private @Wrapped_i32(%self: !trait.claim<@Wrapped[i32]>) {
  %nine = trait.witness @Nine for @Mark[i32]
  trait.return %nine : !trait.claim<@Mark[i32] by @Nine>
}
trait.proof private @W9 {
  %d = trait.derive @Wrapped[i32] from @Wrapped_i32 given()
  trait.return %d : !trait.claim<@Wrapped[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @W9 for @Wrapped[i32]
  // expected-error @below {{may not swap the proof backing claim #trait<application@Mark[i32]>: a coerce compares modulo a proof but may not exchange it for another}}
  %m = trait.project %w[0] : !trait.claim<@Wrapped[i32] by @W9> -> !trait.claim<@Mark[i32] by @Seven>
  %v = trait.method.call %m @Mark[i32]::@value() : () -> i64 by @Seven
  return %v : i64
}
