// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 >/dev/null | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I's header spells @H[i32]::E, which grows at every resolution and has no
// normal form, so @I serves no application. Selecting for @A[i64] reads that
// header, finds it no candidate, names nothing, and chooses @I2.

// QUIET-NOT: error
// CHECK: 9

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E }
trait.impl private @Hg(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E = !trait.proj<@H[tuple<!T>], "E"> }
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.method @value() -> i64 }
trait.impl private @I(%s: !trait.claim<@A[!trait.proj<@H[i32], "E">]>) { trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }
trait.impl private @I2(%s: !trait.claim<@A[i64]>) { trait.method @value() -> i64 { %c = arith.constant 9 : i64 trait.return %c : i64 } }
func.func @main() -> i64 {
  %a = trait.allege @A[i64]
  %v = trait.method.call %a @A[i64]::@value() : () -> i64
  return %v : i64
}
