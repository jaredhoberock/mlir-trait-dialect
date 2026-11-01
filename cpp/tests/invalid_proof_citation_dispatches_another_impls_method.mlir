// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// What an impl returns for a requirement decides which member a body reaches:
// @B_i32's body projects its own requirement @A[i32] and calls @a through it.
// Evidence from @A_i64 there would carry the call to @A_i64's member, so the
// program would answer 64 where it must answer 32. The mismatch is refused
// where it is written.

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>) -> i64
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> members [@B.b]
func.func private @B.b(!trait.claim<@B[!trait.poly<0>]>, !trait.poly<0>) -> i64
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>) -> i64 {
  %c = arith.constant 32 : i64
  return %c : i64
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) members [@A.a -> @A_i64.a] {
}
func.func private @A_i64.a(%self: !trait.claim<@A[i64]>) -> i64 {
  %c = arith.constant 64 : i64
  return %c : i64
}
// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[i32]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
  %r = trait.func.call @A.a(%a) : (!trait.claim<@A[i32]>) -> i64
  return %r : i64
}

func.func @main(%x: i32) -> i64 {
  %c = trait.allege @B[i32]
  %r = trait.func.call @B.b(%c, %x) : (!trait.claim<@B[i32]>, i32) -> i64
  return %r : i64
}
