// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @OnlyI64 applies where its parameter is i64. @p stands over every instance of
// @Vector, and at the declaration the premise reads over the proof's own
// variable and is left to the instances. A witness names one of those
// instances, so the premise is read there.

trait.trait private @Vector[!trait.poly<0>] {
  func.func private @v() -> i64
}
trait.impl private @OnlyI64 for @Vector[!trait.poly<0>]
    where [!trait.poly<0> = i64] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.proof private @p proves @OnlyI64 for @Vector[!trait.poly<0>] given []
func.func @main() -> i64 {
  // expected-error @below {{impl '@OnlyI64' applies where '!trait.poly<0>' = 'i64', and nothing here makes 'i8' and 'i64' one type at '!trait.claim<@Vector[i8] by @p>'}}
  %w = trait.witness @p for @Vector[i8]
  %r = trait.method.call %w @Vector[i8]::@v() : () -> i64 by @p
  return %r : i64
}
