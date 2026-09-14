// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// @I applies where i32 is i64, which it is not. An equality premise takes no
// subproof, so it is read at the application the citation names; a witness
// naming the impl directly names one of those applications.

trait.trait private @T[!trait.poly<0>] {
  func.func private @m() -> i64
}
trait.impl private @I for @T[i32] where [i32 = i64] {
  func.func @m() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
func.func @main() -> i64 {
  // expected-error @below {{impl '@I' applies where 'i32' = 'i64', and nothing here makes 'i32' and 'i64' one type at '!trait.claim<@T[i32] by @I>'}}
  %w = trait.witness @I for @T[i32]
  %r = trait.method.call %w @T[i32]::@m() : () -> i64 by @I
  return %r : i64
}
