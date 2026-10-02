// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B's requirement at @B[i32] is @A[Foo[i32]::Out], and nothing implements @Foo
// at i32, so the projection is settled by nothing. @B_i32 returns @A_i64's
// evidence for it, which nothing the impl reads makes that requirement, so the
// return is refused where it is written.

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Foo_i8(%self: !trait.claim<@Foo[i8]>) { trait.assoc_type @Out = i8 }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> {
  trait.method @value() -> i64
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {}
// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @value() -> i64 {
    %c = arith.constant 13 : i64
    trait.return %c : i64
  }
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}
func.func @main() -> i64 {
  %w = trait.witness @B_i32 for @B[i32]
  %r = trait.method.call %w @B[i32]::@value() : () -> i64 by @B_i32
  return %r : i64
}
