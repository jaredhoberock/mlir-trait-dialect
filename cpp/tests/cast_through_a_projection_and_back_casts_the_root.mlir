// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @p casts the unconditional impl @I from @T[i32] to @T[@A[i32]::Out]. Carrying
// @I on to a further spelling is one cast of the root, not a cast of @p: @q
// coerces the witness of @I through both steps' equalities, there and back, and
// the call it serves runs @I's method.

// CHECK: {{^}}37{{$}}

trait.trait private @T(%s: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @I(%s: !trait.claim<@T[i32]>) {
  trait.method @m() -> i64 {
    %v = arith.constant 37 : i64
    trait.return %v : i64
  }
}
trait.trait private @A(%s: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Out }
trait.impl private @A_i32(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Out = i32 }
trait.proof private @p {
  %w = trait.witness @I for @T[i32]
  %e = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i32 by @A_i32 : !trait.claim<!trait.proj<@A[i32], "Out"> = i32>
  %c = trait.coerce %w : !trait.claim<@T[i32] by @I> to !trait.claim<@T[!trait.proj<@A[i32], "Out">]> via (%e) : (!trait.claim<!trait.proj<@A[i32], "Out"> = i32>)
  trait.return %c : !trait.claim<@T[!trait.proj<@A[i32], "Out">]>
}
trait.proof private @q {
  %w = trait.witness @I for @T[i32]
  %there = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i32 by @A_i32 : !trait.claim<!trait.proj<@A[i32], "Out"> = i32>
  %back = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i32 by @A_i32 : !trait.claim<!trait.proj<@A[i32], "Out"> = i32>
  %c = trait.coerce %w : !trait.claim<@T[i32] by @I> to !trait.claim<@T[i32]> via (%there, %back) : (!trait.claim<!trait.proj<@A[i32], "Out"> = i32>, !trait.claim<!trait.proj<@A[i32], "Out"> = i32>)
  trait.return %c : !trait.claim<@T[i32]>
}
func.func @main() -> i64 {
  %w = trait.witness @q for @T[i32]
  %v = trait.method.call %w @T[i32]::@m() : () -> i64 by @q
  return %v : i64
}
