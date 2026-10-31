// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B_blanket's second premise projects through its first. Two impls of @Foo
// stand at i32, binding Out to i32 and to i64, so the impls around the proof
// do not say what @Foo[i32]::Out is; the witness @forged supplies for the
// first premise does. Read through it, the second premise is @A[i64], and
// @forged supplies @A_i32's evidence for it.

trait.trait private @Foo(!trait.claim<@Foo[!trait.poly<0>]>) assoc {Out = 0}
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) bindings {Out = !trait.poly<0>} {
}
trait.impl private @Foo_i32(%self: !trait.claim<@Foo[i32]>) bindings {Out = i64} {
}
trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>) -> i64
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!trait.poly<0>]>) -> i64
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>) -> i64 {
  %c = arith.constant 32 : i64
  return %c : i64
}
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>, %foo: !trait.claim<@Foo[!trait.poly<0>]>, %a: !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>) members [@B.b -> @B_blanket.b] {
}
func.func private @B_blanket.b(%self: !trait.claim<@B[!trait.poly<0>]>, %foo: !trait.claim<@Foo[!trait.poly<0>]>, %a: !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>) -> i64 {
  %r = trait.func.call @A.a(%a) : (!trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>) -> i64
  return %r : i64
}
trait.proof private @forged {
  %foo = trait.witness @Foo_i32 for @Foo[i32]
  %a = trait.witness @A_i32 for @A[i32]
  // expected-error @below {{premise 1 of impl '@B_blanket' is '!trait.claim<@A[!trait.proj<@Foo[i32], "Out">]>', and the derive supplies '!trait.claim<@A[i32]>'}}
  %d = trait.derive @B[i32] from @B_blanket[i32] given(%foo, %a) : (!trait.claim<@Foo[i32] by @Foo_i32>, !trait.claim<@A[i32] by @A_i32>)
  trait.return %d : !trait.claim<@B[i32]>
}
