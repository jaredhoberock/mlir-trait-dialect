// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @B's second requirement projects through @Foo, its first, which @forged
// discharges with a proof of @Foo_any -- an impl binding Out to its own
// variable. That subproof is the evidence at the requirement's index, so the
// obligation @B_blanket states is @A of the variable itself. @forged cites
// @A_i64 for it, evidence of @A at one argument, and the two claims are
// compared where the proof stands.

trait.trait private @Foo[!trait.poly<0>] { trait.assoc_type @Out }
trait.impl private @Foo_any for @Foo[!trait.poly<0>] { trait.assoc_type @Out = !trait.poly<0> }
trait.trait private @A[!trait.poly<0>] { trait.method @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@Foo[!trait.poly<0>], @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]] { trait.method @b() -> i64 }
trait.impl private @A_i32 for @A[i32] {
  trait.method @a() -> i64 {
    %c = arith.constant 32 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_i64 for @A[i64] {
  trait.method @a() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  trait.method @b() -> i64 {
    %s = trait.assume self : !trait.claim<@B[!trait.poly<0>]>
    %a = trait.project %s[1] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.proj<@Foo[!trait.poly<0>], "Out">]>
    %r = trait.method.call %a @A[!trait.proj<@Foo[!trait.poly<0>], "Out">]::@a() : () -> i64
    trait.return %r : i64
  }
}
trait.proof private @Foo_any_p proves @Foo_any[!trait.poly<0> = !trait.poly<0>] for @Foo[!trait.poly<0>] given []
// expected-error @below {{proof @A_i64 proves '!trait.claim<@A[i64]>', which does not discharge the obligation '!trait.claim<@A[!trait.poly<0>]>'}}
trait.proof private @forged proves @B_blanket[!trait.poly<0> = !trait.poly<0>] for @B[!trait.poly<0>] given [@Foo_any_p, @A_i64]
func.func @main() -> i64 {
  %w = trait.witness @forged for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @forged
  return %r : i64
}
