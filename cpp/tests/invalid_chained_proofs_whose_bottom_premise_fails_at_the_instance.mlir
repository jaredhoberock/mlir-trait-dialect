// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @px cites @pu, which cites @pv, which stands over @OnlyI64 -- an impl
// applying only where its parameter is i64. Each subproof's claim is the
// obligation it discharges at the application the citation names, so
// witnessing @px at i8 carries i8 to the bottom of the chain and the premise
// is read there. Read at each proof's own declaration instead, the premise
// stands over a variable at every level and defers all the way down.

trait.trait private @V[!trait.poly<0>] { func.func private @v() -> i64 }
trait.trait private @U[!trait.poly<0>] where [@V[!trait.poly<0>]] { func.func private @u() -> i64 }
trait.trait private @X[!trait.poly<0>] where [@U[!trait.poly<0>]] { func.func private @x() -> i64 }
trait.impl private @OnlyI64 for @V[!trait.poly<0>] where [!trait.poly<0> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @U_blanket for @U[!trait.poly<0>] {
  func.func @u() -> i64 {
    %s = trait.assume @U[!trait.poly<0>]
    %a = trait.project %s[0] : !trait.claim<@U[!trait.poly<0>]> -> !trait.claim<@V[!trait.poly<0>]>
    %r = trait.method.call %a @V[!trait.poly<0>]::@v() : () -> i64
    return %r : i64
  }
}
trait.impl private @X_blanket for @X[!trait.poly<0>] {
  func.func @x() -> i64 {
    %s = trait.assume @X[!trait.poly<0>]
    %a = trait.project %s[0] : !trait.claim<@X[!trait.poly<0>]> -> !trait.claim<@U[!trait.poly<0>]>
    %r = trait.method.call %a @U[!trait.poly<0>]::@u() : () -> i64
    return %r : i64
  }
}
trait.proof private @pv proves @OnlyI64 for @V[!trait.poly<0>] given []
trait.proof private @pu proves @U_blanket for @U[!trait.poly<0>] given [@pv]
trait.proof private @px proves @X_blanket for @X[!trait.poly<0>] given [@pu]
func.func @main() -> i64 {
  %w = trait.witness @px for @X[i8]
  // expected-error @below {{impl '@OnlyI64' applies where '!trait.poly<0>' = 'i64', and nothing here makes 'i8' and 'i64' one type at '!trait.claim<@V[i8]>'}}
  %r = trait.method.call %w @X[i8]::@x() : () -> i64 by @px
  return %r : i64
}
