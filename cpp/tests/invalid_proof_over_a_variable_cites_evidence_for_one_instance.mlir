// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// An impl written over a type variable stands for every instance of it, so the
// evidence it returns for its requirement must stand for every instance too.
// @A_i64 is evidence at one instance, and a witness of @B_blanket at another
// one would dispatch its requirement through it: @B[i32]::@b would call
// @A_i64's method where @A_i32's is the impl for i32.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
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
// expected-error@+1 {{returns '!trait.claim<@A[i64] by @A_i64>' for requirement 0, which trait '@B' states as '!trait.claim<@A[!trait.poly<0>]>'}}
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @b(%x: !trait.poly<0>) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
    %r = trait.method.call %a @A[!trait.poly<0>]::@a() : () -> i64
    trait.return %r : i64
  }
  %a = trait.witness @A_i64 for @A[i64]
  trait.return %a : !trait.claim<@A[i64] by @A_i64>
}

// -----

// The evidence an impl over a variable does take: a derive of a blanket impl
// at the requirement's own spelling, which stands for every instance of it.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  trait.method @b(!trait.poly<0>) -> i64
}
trait.impl private @A_blanket(%self: !trait.claim<@A[!trait.poly<0>]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_blanket(%self: !trait.claim<@B[!trait.poly<0>]>) {
  trait.method @b(%x: !trait.poly<0>) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
    %r = trait.method.call %a @A[!trait.poly<0>]::@a() : () -> i64
    trait.return %r : i64
  }
  %a = trait.derive @A[!trait.poly<0>] from @A_blanket[!trait.poly<0>] given()
  trait.return %a : !trait.claim<@A[!trait.poly<0>]>
}
