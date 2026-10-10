// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B_blanket's second premise projects through its first, which @forged
// supplies with a proof of @Foo_any -- an impl binding Out to its own
// variable -- at i32. That witness is the evidence at the premise's index, so
// the premise @B_blanket states is @A of i32. @forged supplies @A_i64's
// evidence for it, evidence of @A at another argument, and the two claims are
// compared where the derive stands.

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out = !trait.poly<0> }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) { trait.method @b() -> i64 }
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
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>, %foo: !trait.claim<@Foo[!trait.poly<0>]>, %a: !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>) {
  trait.method @b() -> i64 {
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
}
trait.proof private @Foo_any_p {
  %d = trait.derive @Foo[i32] from @Foo_any given()
  trait.return %d : !trait.claim<@Foo[i32]>
}
trait.proof private @forged {
  %foo = trait.witness @Foo_any_p for @Foo[i32]
  %a = trait.witness @A_i64 for @A[i64]
  // expected-error @below {{premise 1 of impl '@B_blanket' is '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>', and the derive supplies '!trait.claim<@A[i64]>'}}
  %d = trait.derive @B[i32] from @B_blanket given(%foo, %a) : (!trait.claim<@Foo[i32] by @Foo_any_p>, !trait.claim<@A[i64] by @A_i64>)
  trait.return %d : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
