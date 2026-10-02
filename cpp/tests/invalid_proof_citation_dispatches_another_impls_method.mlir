// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// What an impl returns for a requirement decides which method a body reaches:
// @B_i32's body projects its own requirement @A[i32] and calls @a through it.
// Evidence from @A_i64 there would carry the call to @A_i64's method, so the
// program would answer 64 where it must answer 32. The mismatch is refused
// where it is written.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {
  trait.method @a() -> i64
}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  trait.method @b(!trait.poly<0>) -> i64
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 32 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[i32]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @b(%x: i32) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
    %r = trait.method.call %a @A[i32]::@a() : () -> i64
    trait.return %r : i64
  }
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}

func.func @main(%x: i32) -> i64 {
  %c = trait.allege @B[i32]
  %r = trait.method.call %c @B[i32]::@b(%x) : (i32) -> i64
  return %r : i64
}
