// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Two calls of @g supply @Tr[i32] for different reasons: one through @One, whose
// member answers 7, one through @Two, whose member answers 9. An instance is
// named by the evidence it is made with, so the calls reach two instances of @g
// at one type argument, and each runs the impl its own proof selects.

// CHECK: {{^}}16{{$}}

!T = !trait.poly<0>
trait.trait private @Tr(!trait.claim<@Tr[!T]>) members [@Tr.value]
func.func private @Tr.value(!trait.claim<@Tr[!T]>) -> i64
trait.impl private @One(%self: !trait.claim<@Tr[i32]>) members [@Tr.value -> @One.value] {
}
func.func private @One.value(%self: !trait.claim<@Tr[i32]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Two(%self: !trait.claim<@Tr[i32]>) members [@Tr.value -> @Two.value] {
}
func.func private @Two.value(%self: !trait.claim<@Tr[i32]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
func.func private @g(%c: !trait.claim<@Tr[!T]>) -> i64 {
  %v = trait.func.call @Tr.value(%c) : (!trait.claim<@Tr[!T]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %one = trait.witness @One for @Tr[i32]
  %two = trait.witness @Two for @Tr[i32]
  %a = trait.func.call @g(%one) : (!trait.claim<@Tr[i32] by @One>) -> i64
  %b = trait.func.call @g(%two) : (!trait.claim<@Tr[i32] by @Two>) -> i64
  %sum = arith.addi %a, %b : i64
  return %sum : i64
}
