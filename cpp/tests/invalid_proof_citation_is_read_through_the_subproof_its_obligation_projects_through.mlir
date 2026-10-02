// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B_blanket's second premise projects through its first. Two impls of @Foo
// stand at i32, binding Out to i32 and to i64, so the impls around the proof
// do not say what @Foo[i32]::Out is; the witness @forged supplies for the
// first premise does. Read through it, the second premise is @A[i64], and
// @forged supplies @A_i32's evidence for it.

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out = !trait.poly<0> }
trait.impl private @Foo_i32(%self: !trait.claim<@Foo[i32]>) { trait.assoc_type @Out = i64 }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) { trait.method @b() -> i64 }
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 32 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>, %foo: !trait.claim<@Foo[!trait.poly<0>]>, %a: !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>) {
  trait.method @b() -> i64 {
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
}
trait.proof private @forged {
  %foo = trait.witness @Foo_i32 for @Foo[i32]
  %a = trait.witness @A_i32 for @A[i32]
  // expected-error @below {{premise 1 of impl '@B_blanket' is '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>', and the derive supplies '!trait.claim<@A[i32]>'}}
  %d = trait.derive @B[i32] from @B_blanket given(%foo, %a) : (!trait.claim<@Foo[i32] by @Foo_i32>, !trait.claim<@A[i32] by @A_i32>)
  trait.return %d : !trait.claim<@B[i32]>
}
