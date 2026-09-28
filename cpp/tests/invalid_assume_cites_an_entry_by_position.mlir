// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A position past the end of the where clause names no entry.

!T = !trait.poly<0>
trait.trait private @B[!T] { func.func private @b(!T) -> i64 }
trait.trait private @A[!T] { func.func private @a(!T) -> i64 }
trait.impl private @A_gen for @A[!T] where [@B[!T]] {
  func.func @a(%x: !T) -> i64 {
    // expected-error @below {{cites where-clause entry 1, but the enclosing declaration's where clause has 1 entries}}
    %b = trait.assume 1 : !trait.claim<@B[!T]>
    %c = arith.constant 0 : i64
    return %c : i64
  }
}

// -----

// The result type spells a claim other than the entry at the position.

!T = !trait.poly<0>
trait.trait private @B[!T] { func.func private @b(!T) -> i64 }
trait.trait private @C[!T] { func.func private @c(!T) -> i64 }
trait.trait private @A[!T] { func.func private @a(!T) -> i64 }
trait.impl private @A_gen for @A[!T] where [@B[!T], @C[!T]] {
  func.func @a(%x: !T) -> i64 {
    // expected-error @below {{the cited entry states '!trait.claim<@B[!trait.poly<0>]>', but the result type spells '!trait.claim<@C[!trait.poly<0>]>'}}
    %b = trait.assume 0 : !trait.claim<@C[!T]>
    %c = arith.constant 0 : i64
    return %c : i64
  }
}

// -----

// The result type spells a claim other than the declaration's self application.

!T = !trait.poly<0>
trait.trait private @A[!T] {
  func.func private @a(!T) -> i64
  func.func @twice(%x: !T) -> i64 {
    // expected-error @below {{the cited entry states '!trait.claim<@A[!trait.poly<0>]>', but the result type spells '!trait.claim<@A[i64]>'}}
    %s = trait.assume self : !trait.claim<@A[i64]>
    %c = arith.constant 0 : i64
    return %c : i64
  }
}

// -----

// An equality entry states an equality claim, not an application.

!T = !trait.poly<0>
trait.trait private @B[!T] { trait.assoc_type @Out }
trait.trait private @A[!T] { func.func private @a(!T) -> i64 }
trait.impl private @A_gen for @A[!T] where [!trait.proj<@B[!T], "Out"> = i64] {
  func.func @a(%x: !T) -> i64 {
    // expected-error @below {{the cited entry states '!trait.claim<!trait.proj<@B[!trait.poly<0>], "Out"> = i64>', but the result type spells '!trait.claim<@B[!trait.poly<0>]>'}}
    %b = trait.assume 0 : !trait.claim<@B[!T]>
    %c = arith.constant 0 : i64
    return %c : i64
  }
}

// -----

// A free function is a method of no declaration, so it has no entry to cite.

!T = !trait.poly<0>
trait.trait private @B[!T] { func.func private @b(!T) -> i64 }
func.func @f(%c: !trait.claim<@B[!T]>, %x: !T) -> i64 {
  // expected-error @below {{cites an entry of the declaration its function is a method of, but '@f' is a method of no trait or impl}}
  %b = trait.assume 0 : !trait.claim<@B[!T]>
  %r = trait.method.call %b @B[!T]::@b(%x) : (!T) -> i64
  return %r : i64
}
