// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s 2>&1 | FileCheck %s

// @B's requirement projects through @Foo, which @B does not require, so the
// impl's own bindings and where entries do not say what that projection is.
// @B_blanket returns a witness of @A_i64 for it: the return check reads the
// requirement through what the impl holds and nothing else -- not the impls
// standing around it -- so it cannot equate @A[i64] with
// @A[@Foo[T]::Out] and refuses the evidence where it is returned, before any
// instance is cut.

// CHECK: error: 'trait.impl' op returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>'

trait.trait private @Foo(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @Foo_any(%self: !trait.claim<@Foo[!trait.poly<0>]>) { trait.assoc_type @Out = !trait.poly<0> }
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]> { trait.method @b() -> i64 }
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @b() -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
  %a64 = trait.witness @A_i64 for @A[i64]
  trait.return %a64 : !trait.claim<@A[i64] by @A_i64>
}
trait.proof private @forged {
  %d = trait.derive @B[i32] from @B_blanket[i32] given()
  trait.return %d : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
