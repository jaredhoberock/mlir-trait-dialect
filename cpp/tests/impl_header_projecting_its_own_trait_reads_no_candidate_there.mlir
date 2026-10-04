// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I0's header projects through its own trait, at @H[tuple<i32>]::E. Reading
// that header asks for @H[tuple<i32>]'s candidates, among them @I0's header
// again: there the application whose candidates are being read has none, so
// @I0 is no candidate of @H[tuple<i32>], @I1 binds E = i32, and @I0 serves
// @H[i32].

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) { trait.assoc_type @E  trait.method @value() -> i64 }
trait.impl private @I0(%s: !trait.claim<@H[!trait.proj<@H[tuple<i32>], "E">]>) { trait.assoc_type @E = i64  trait.method @value() -> i64 { %c = arith.constant 7 : i64 trait.return %c : i64 } }
trait.impl private @I1(%s: !trait.claim<@H[tuple<i32>]>) { trait.assoc_type @E = i32  trait.method @value() -> i64 { %c = arith.constant 8 : i64 trait.return %c : i64 } }
func.func @main() -> i64 {
  %a = trait.allege @H[i32]
  %v = trait.method.call %a @H[i32]::@value() : () -> i64
  return %v : i64
}
