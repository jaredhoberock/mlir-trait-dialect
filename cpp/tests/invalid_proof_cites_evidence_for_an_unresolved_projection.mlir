// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @B's requirement is spelled over a projection, and two impls of @Foo bind
// @Foo[i32] -- both to i32 -- so nothing the impl reads resolves it. Evidence
// read against a requirement nothing can settle is evidence nothing checked:
// @B_i32 returns @A_i64's evidence where the projection denotes i32. The
// return is refused where it is written, rather than the call through the
// requirement being dispatched to @A_i64's method.

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out = i32 }
trait.impl private @Foo_i32(%self: !trait.claim<@Foo[i32]>) { trait.assoc_type @Out = i32 }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> {
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
// expected-error @below {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>'}}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @b(%x: i32) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[i32], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}
func.func @main(%x: i32) -> i64 {
  %w = trait.witness @B_i32 for @B[i32]
  %r = trait.method.call %w @B[i32]::@b(%x) : (i32) -> i64 by @B_i32
  return %r : i64
}
