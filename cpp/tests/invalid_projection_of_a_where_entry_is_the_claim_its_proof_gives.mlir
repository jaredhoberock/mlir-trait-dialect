// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A projection of a where entry is the claim the source's proof gives its
// derive there, proven or not. @p gives @B_impl's entry an unproven claim of
// @A[i32]: the derive @A_i32 is its evidence, so a projection naming @A_alt, a
// proof of the same application, names evidence the proof was never given.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) { trait.method @m() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @m() -> i64 { %c = arith.constant 7 : i64  trait.return %c : i64 }
}
trait.impl private @A_alt(%self: !trait.claim<@A[i32]>) {
  trait.method @m() -> i64 { %c = arith.constant 8 : i64  trait.return %c : i64 }
}
trait.impl private @B_impl(%self: !trait.claim<@B[!T]>, %a: !trait.claim<@A[!T]>) {}
trait.proof private @p {
  %a = trait.derive @A[i32] from @A_i32 given()
  %d = trait.derive @B[i32] from @B_impl[i32] given(%a) : (!trait.claim<@A[i32]>)
  trait.return %d : !trait.claim<@B[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @p for @B[i32]
  // expected-error @below {{type mismatch: expected '!trait.claim<@A[i32]>' but found '!trait.claim<@A[i32] by @A_alt>'}}
  %x = trait.project %w[0] : !trait.claim<@B[i32] by @p> -> !trait.claim<@A[i32] by @A_alt>
  %r = trait.method.call %x @A[i32]::@m() : () -> i64 by @A_alt
  return %r : i64
}

// -----

// The entry given a coercion that drops its input's proof.

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {}
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {}
trait.impl private @A_alt(%self: !trait.claim<@A[i32]>) {}
trait.impl private @B_impl(%self: !trait.claim<@B[!T]>, %a: !trait.claim<@A[!T]>) {}
trait.proof private @p {
  %a = trait.witness @A_i32 for @A[i32]
  %u = trait.coerce %a : !trait.claim<@A[i32] by @A_i32> to !trait.claim<@A[i32]>
  %d = trait.derive @B[i32] from @B_impl[i32] given(%u) : (!trait.claim<@A[i32]>)
  trait.return %d : !trait.claim<@B[i32]>
}
func.func @f() {
  %w = trait.witness @p for @B[i32]
  // expected-error @below {{type mismatch: expected '!trait.claim<@A[i32]>' but found '!trait.claim<@A[i32] by @A_alt>'}}
  %x = trait.project %w[0] : !trait.claim<@B[i32] by @p> -> !trait.claim<@A[i32] by @A_alt>
  return
}
