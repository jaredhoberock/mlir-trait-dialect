// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A trait's requirement may be an equality: @Same requires Same[S]::A = i64,
// and its impl returns the witness of its own binding. A generic function
// projects the equality off its claim and coerces through it; the instance at
// i32 runs.

// CHECK: {{^}}5{{$}}

!S = !trait.poly<0>
!T = !trait.poly<1>
trait.trait private @Same(%self: !trait.claim<@Same[!S]>) -> !trait.claim<!trait.proj<@Same[!S], "A"> = i64> {
  trait.assoc_type @A
}
trait.impl private @Same_i32(%self: !trait.claim<@Same[i32]>) {
  trait.assoc_type @A = i64
  %a = trait.witness proj_resolve !trait.proj<@Same[i32], "A"> resolves i64 by @Same_i32 : !trait.claim<!trait.proj<@Same[i32], "A"> = i64>
  trait.return %a : !trait.claim<!trait.proj<@Same[i32], "A"> = i64>
}
func.func private @read(%s: !trait.claim<@Same[!trait.poly<0>]>, %v: !trait.proj<@Same[!trait.poly<0>], "A">) -> i64 {
  %e = trait.project %s[0] : !trait.claim<@Same[!trait.poly<0>]> -> !trait.claim<!trait.proj<@Same[!trait.poly<0>], "A"> = i64>
  %r = trait.coerce %v : !trait.proj<@Same[!trait.poly<0>], "A"> to i64 via (%e) : (!trait.claim<!trait.proj<@Same[!trait.poly<0>], "A"> = i64>)
  return %r : i64
}
func.func @main() -> i64 {
  %x = arith.constant 5 : i64
  %s = trait.witness @Same_i32 for @Same[i32]
  %e = trait.project %s[0] : !trait.claim<@Same[i32] by @Same_i32> -> !trait.claim<!trait.proj<@Same[i32], "A"> = i64>
  %v = trait.coerce %x : i64 to !trait.proj<@Same[i32], "A"> via (%e) : (!trait.claim<!trait.proj<@Same[i32], "A"> = i64>)
  %r = trait.func.call @read(%s, %v) : (!trait.claim<@Same[i32] by @Same_i32>, !trait.proj<@Same[i32], "A">) -> i64
  return %r : i64
}
