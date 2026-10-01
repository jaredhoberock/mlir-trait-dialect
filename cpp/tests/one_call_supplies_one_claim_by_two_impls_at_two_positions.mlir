// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One call supplies @Tr[i32] at both of @g's claim parameters, through @One
// (7) at the first and @Two (9) at the second. The two parameters spell one
// claim, so no respelling keyed by that claim can say which proof either means;
// each parameter of the instance carries the evidence supplied at its own
// position, and each method call runs the impl that evidence selects.

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
func.func private @g(%a: !trait.claim<@Tr[!T]>, %b: !trait.claim<@Tr[!T]>) -> i64 {
  %x = trait.method.call %a @Tr[!T]::@value() : () -> i64
  %y = trait.method.call %b @Tr[!T]::@value() : () -> i64
  %s = arith.addi %x, %y : i64
  return %s : i64
}
func.func @main() -> i64 {
  %one = trait.witness @One for @Tr[i32]
  %two = trait.witness @Two for @Tr[i32]
  %r = trait.func.call @g(%one, %two) : (!trait.claim<@Tr[i32] by @One>, !trait.claim<@Tr[i32] by @Two>) -> i64
  return %r : i64
}
