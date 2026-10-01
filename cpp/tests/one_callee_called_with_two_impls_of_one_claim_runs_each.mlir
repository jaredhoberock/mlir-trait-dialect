// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Two calls of @g supply @Tr[i32] for different reasons: one through @One, whose
// method answers 7, one through @Two, whose method answers 9. An instance is
// named by the evidence it is made with, so the calls reach two instances of @g
// at one type argument, and each runs the impl its own proof selects.

// CHECK: {{^}}16{{$}}

!T = !trait.poly<0>
trait.trait private @Tr[!T] { trait.method @value() -> i64 }
trait.impl private @One for @Tr[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Two for @Tr[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
func.func private @g(%c: !trait.claim<@Tr[!T]>) -> i64 {
  %v = trait.method.call %c @Tr[!T]::@value() : () -> i64
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
