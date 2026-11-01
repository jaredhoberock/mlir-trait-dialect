// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B's requirement is spelled over a projection, and two impls of @Foo bind
// @Foo[i32] -- both to i32 -- so nothing the impl reads resolves it. Evidence
// read against a requirement nothing can settle is evidence nothing checked:
// @B_i32 returns @A_i64's evidence where the projection denotes i32. The
// return is refused where it is written, rather than the call through the
// requirement being dispatched to @A_i64's member.

trait.trait private @Foo(!trait.claim<@Foo[!trait.poly<0>]>) assoc {Out = 0}
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) bindings {Out = i32} {
}
trait.impl private @Foo_i32(%self: !trait.claim<@Foo[i32]>) bindings {Out = i32} {
}
trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>) -> i64
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> members [@B.b]
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
// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>
  %r = trait.func.call @A.a(%a) : (!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>) -> i64
  return %r : i64
}
func.func @main(%x: i32) -> i64 {
  %w = trait.witness @B_i32 for @B[i32]
  %r = trait.func.call @B.b(%w, %x) : (!trait.claim<@B[i32] by @B_i32>, i32) -> i64
  return %r : i64
}
