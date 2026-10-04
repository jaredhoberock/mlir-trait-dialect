// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @H[i1]::E is bound to @H[i2]::E, and so on to @H[i101]::E, which is i32: a
// finite chain of one hundred and one projection steps, short of the depth
// limit, so the demand settles to @A[i32] and @A_i32's method runs.

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.method @value() -> i64 }
trait.impl private @A_i32(%s: !trait.claim<@A[i32]>) { trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }
// REPEAT 1 100: trait.impl private @H{k}(%s: !trait.claim<@H[i{k}]>) { trait.assoc_type @E = !trait.proj<@H[i{k+1}], "E"> }
trait.impl private @H101(%s: !trait.claim<@H[i101]>) { trait.assoc_type @E = i32 }
func.func @main() -> i64 {
  %a = trait.allege @A[!trait.proj<@H[i1], "E">]
  %v = trait.method.call %a @A[!trait.proj<@H[i1], "E">]::@value() : () -> i64
  return %v : i64
}
