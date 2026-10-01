// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @px cites @pu, which cites @pv, which stands over @OnlyI64 -- an impl
// applying only where its parameter is i64. @pv's claim leaves that premise
// over its own variable, and nothing above it reads a premise of a proof it
// cites, so the refusal lands at @pv. The two proofs above it and the witness
// at i8 each name a declaration that carries to the claim they stand on, which
// is the whole of what they assert.

trait.trait private @V[!trait.poly<0>] { trait.method @v() -> i64 }
trait.trait private @U[!trait.poly<0>] where [@V[!trait.poly<0>]] { trait.method @u() -> i64 }
trait.trait private @X[!trait.poly<0>] where [@U[!trait.poly<0>]] { trait.method @x() -> i64 }
trait.impl private @OnlyI64 for @V[!trait.poly<0>] where [!trait.poly<0> = i64] {
  trait.method @v() -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
}
trait.impl private @U_blanket for @U[!trait.poly<0>] {
  trait.method @u() -> i64 {
    %s = trait.assume self : !trait.claim<@U[!trait.poly<0>]>
    %a = trait.project %s[0] : !trait.claim<@U[!trait.poly<0>]> -> !trait.claim<@V[!trait.poly<0>]>
    %r = trait.method.call %a @V[!trait.poly<0>]::@v() : () -> i64
    trait.return %r : i64
  }
}
trait.impl private @X_blanket for @X[!trait.poly<0>] {
  trait.method @x() -> i64 {
    %s = trait.assume self : !trait.claim<@X[!trait.poly<0>]>
    %a = trait.project %s[0] : !trait.claim<@X[!trait.poly<0>]> -> !trait.claim<@U[!trait.poly<0>]>
    %r = trait.method.call %a @U[!trait.poly<0>]::@u() : () -> i64
    trait.return %r : i64
  }
}
// expected-error @below {{a proof states its impl's premises at its own claim; one the claim leaves open is stated at the instance instead: '!trait.poly<0>' = 'i64' reads '!trait.poly<0>' = 'i64' at '!trait.claim<@V[!trait.poly<0>] by @pv>'}}
trait.proof private @pv proves @OnlyI64[!trait.poly<0> = !trait.poly<0>] for @V[!trait.poly<0>] given [unit]
trait.proof private @pu proves @U_blanket[!trait.poly<0> = !trait.poly<0>] for @U[!trait.poly<0>] given [@pv]
trait.proof private @px proves @X_blanket[!trait.poly<0> = !trait.poly<0>] for @X[!trait.poly<0>] given [@pu]
func.func @main() -> i64 {
  %w = trait.witness @px for @X[i8]
  %r = trait.method.call %w @X[i8]::@x() : () -> i64 by @px
  return %r : i64
}
