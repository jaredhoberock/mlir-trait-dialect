// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A call of this dialect names its callee in its module and lowers to a
// func.call where it stands, which names a function of its own symbol table
// alone. Inside a symbol table other than its module or a trait or impl
// declaration -- whose methods are cut into the module before a call they hold
// lowers -- no lowering reaches the callee, so the call is refused where it
// stands.

func.func @f(%x: i64) -> i64 { return %x : i64 }
gpu.module @g {
  func.func @use(%x: i64) -> i64 {
    // expected-error @below {{'trait.func.call' op stands inside 'gpu.module', a symbol table below the module its callee is named in}}
    %r = trait.func.call @f(%x) : (i64) -> i64
    return %r : i64
  }
}

// -----

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {
  trait.method @m() -> i64 { %v = arith.constant 32 : i64  trait.return %v : i64 }
}
gpu.module @g {
  func.func @use() -> i64 {
    %w = trait.witness @T_i32 for @T[i32]
    %c = trait.coerce %w : !trait.claim<@T[i32] by @T_i32> to !trait.claim<@T[i32]>
    // expected-error @below {{'trait.method.call' op stands inside 'gpu.module', a symbol table below the module its callee is named in}}
    %v = trait.method.call %c @T[i32]::@m() : () -> i64
    return %v : i64
  }
}
