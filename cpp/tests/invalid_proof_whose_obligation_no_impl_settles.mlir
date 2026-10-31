// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B's requirement at @B[i32] is @A[Foo[i32]::Out], and nothing implements @Foo
// at i32, so the projection is settled by nothing. @B_i32 returns @A_i64's
// evidence for it, which nothing the impl reads makes that requirement, so the
// return is refused where it is written.

trait.trait private @Foo(!trait.claim<@Foo[!trait.poly<0>]>) assoc {Out = 0}
trait.impl private @Foo_i8(%self: !trait.claim<@Foo[i8]>) bindings {Out = i8} {
}
trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>)
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> members [@B.value]
func.func private @B.value(!trait.claim<@B[!trait.poly<0>]>) -> i64
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
}
// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.value -> @B_i32.value] {
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}
func.func private @B_i32.value(%self: !trait.claim<@B[i32]>) -> i64 {
  %c = arith.constant 13 : i64
  return %c : i64
}
func.func @main() -> i64 {
  %w = trait.witness @B_i32 for @B[i32]
  %r = trait.func.call @B.value(%w) : (!trait.claim<@B[i32] by @B_i32>) -> i64
  return %r : i64
}
