// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A proof written over a type variable stands for every instance of it, so what
// discharges its obligation must stand for every instance too. @A_i64 is
// evidence at one instance, and a witness of this proof at another one would
// dispatch @B_blanket's requirement through it: @B[i32]::@b would call
// @A_i64's method where @A_i32's is the impl for i32.

trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {
  func.func private @b(!trait.poly<0>) -> i64
}
trait.impl private @A_i32 for @A[i32] {
  func.func @a() -> i64 {
    %c = arith.constant 32 : i64
    return %c : i64
  }
}
trait.impl private @A_i64 for @A[i64] {
  func.func @a() -> i64 {
    %c = arith.constant 64 : i64
    return %c : i64
  }
}
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  func.func @b(%x: !trait.poly<0>) -> i64 {
    %s = trait.assume @B[!trait.poly<0>]
    %a = trait.project %s[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
    %r = trait.method.call %a @A[!trait.poly<0>]::@a() : () -> i64
    return %r : i64
  }
}
// expected-error@+1 {{proof @A_i64 proves '!trait.claim<@A[i64]>', which does not discharge the obligation '!trait.claim<@A[!trait.poly<0>]>'}}
trait.proof private @forged proves @B_blanket for @B[!trait.poly<0>] given [@A_i64]

// -----

// The evidence a proof over a variable does take: a proof of a blanket impl,
// standing over a variable of its own, rebuilds the obligation at whatever
// that obligation spells.

trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {
  func.func private @b(!trait.poly<0>) -> i64
}
trait.impl private @A_blanket for @A[!trait.poly<1>] {
  func.func @a() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
trait.impl private @B_blanket for @B[!trait.poly<0>] {
  func.func @b(%x: !trait.poly<0>) -> i64 {
    %s = trait.assume @B[!trait.poly<0>]
    %a = trait.project %s[0] : !trait.claim<@B[!trait.poly<0>]> -> !trait.claim<@A[!trait.poly<0>]>
    %r = trait.method.call %a @A[!trait.poly<0>]::@a() : () -> i64
    return %r : i64
  }
}
trait.proof private @a_stands proves @A_blanket for @A[!trait.poly<1>] given []
trait.proof private @stands proves @B_blanket for @B[!trait.poly<0>] given [@a_stands]
