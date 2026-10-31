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
trait.trait private @H(!trait.claim<@H[!T]>) assoc {E = 0} members [@H.value]
func.func private @H.value(!trait.claim<@H[!T]>) -> i64
trait.impl private @I0(%s: !trait.claim<@H[!trait.proj<@H[tuple<i32>], "E">]>) bindings {E = i64} members [@H.value -> @I0.value] {
}
func.func private @I0.value(%s: !trait.claim<@H[!trait.proj<@H[tuple<i32>], "E">]>) -> i64 {
  %c = arith.constant 7 : i64 return %c : i64
}
trait.impl private @I1(%s: !trait.claim<@H[tuple<i32>]>) bindings {E = i32} members [@H.value -> @I1.value] {
}
func.func private @I1.value(%s: !trait.claim<@H[tuple<i32>]>) -> i64 {
  %c = arith.constant 8 : i64 return %c : i64
}
func.func @main() -> i64 {
  %a = trait.allege @H[i32]
  %v = trait.func.call @H.value(%a) : (!trait.claim<@H[i32]>) -> i64
  return %v : i64
}
