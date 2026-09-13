// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// What a citation discharges decides which method a body reaches: @B_i32's
// body projects its own requirement @A[i32] and calls @a through it. A citation
// of @A_i64 there would carry the call to @A_i64's method, so the program would
// answer 64 where it must answer 32. The mismatch is refused where it is
// written.

trait.trait private @A[!trait.poly<0>] {
  func.func private @a() -> i64
}
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
trait.impl private @B_i32 for @B[i32] {
  func.func @b(%x: i32) -> i64 {
    %s = trait.assume @B[i32]
    %a = trait.project %s[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
    %r = trait.method.call %a @A[i32]::@a() : () -> i64
    return %r : i64
  }
}

// expected-error @below {{proof @A_i64 proves '!trait.claim<@A[i64]>', which does not discharge the obligation '!trait.claim<@A[i32]>'}}
trait.proof private @forged proves @B_i32 for @B[i32] given [@A_i64]

func.func @main(%x: i32) -> i64 {
  %c = trait.allege @B[i32]
  %r = trait.method.call %c @B[i32]::@b(%x) : (i32) -> i64
  return %r : i64
}
