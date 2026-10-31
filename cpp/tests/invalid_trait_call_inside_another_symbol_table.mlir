// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A symbol this dialect names resolves in the nearest symbol table around the
// op naming it, as a func.call's callee does. A call standing inside a
// gpu.module names that gpu.module's functions alone, so a callee, or a
// trait, declared only in the module around it is unresolved there.

func.func @f(%x: i64) -> i64 { return %x : i64 }
gpu.module @g {
  func.func @use(%x: i64) -> i64 {
    // expected-error @below {{'trait.func.call' op 'f' does not refer to a valid func.func}}
    %r = trait.func.call @f(%x) : (i64) -> i64
    return %r : i64
  }
}

// -----

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!trait.poly<0>]>) -> i64
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) members [@T.m -> @T_i32.m] {
}
func.func private @T_i32.m(%self: !trait.claim<@T[i32]>) -> i64 {
  %v = arith.constant 32 : i64  return %v : i64
}
gpu.module @g {
  func.func @use() -> i64 {
    // expected-error @below {{cannot find trait '@T'}}
    %w = trait.witness @T_i32 for @T[i32]
    %c = trait.coerce %w : !trait.claim<@T[i32] by @T_i32> to !trait.claim<@T[i32]>
    %v = trait.func.call @T.m(%c) : (!trait.claim<@T[i32]>) -> i64
    return %v : i64
  }
}
