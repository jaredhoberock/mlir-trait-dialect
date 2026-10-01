// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @P_one and @P_two both prove @Tr[i32] through @Tr_impl, and discharge its
// where clause with different impls of @Mark[i32]: @Mark_one (7) and @Mark_two
// (9). A method instance is named by its receiver's proof, and through it by
// every subproof, so the two calls reach two instances of @Tr_impl's method,
// each reading the subproof its own receiver names.

// CHECK: {{^}}16{{$}}

!T = !trait.poly<0>
trait.trait private @Mark[!T] { func.func private @value() -> i64 }
trait.trait private @Tr[!T] { func.func private @value() -> i64 }
trait.impl private @Mark_one for @Mark[i32] {
  func.func @value() -> i64 {
    %v = arith.constant 7 : i64
    return %v : i64
  }
}
trait.impl private @Mark_two for @Mark[i32] {
  func.func @value() -> i64 {
    %v = arith.constant 9 : i64
    return %v : i64
  }
}
trait.impl private @Tr_impl for @Tr[i32] where [@Mark[i32]] {
  func.func @value() -> i64 {
    %m = trait.assume 0 : !trait.claim<@Mark[i32]>
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    return %v : i64
  }
}
trait.proof private @P_one proves @Tr_impl[] for @Tr[i32] given [@Mark_one]
trait.proof private @P_two proves @Tr_impl[] for @Tr[i32] given [@Mark_two]
func.func @main() -> i64 {
  %one = trait.witness @P_one for @Tr[i32]
  %two = trait.witness @P_two for @Tr[i32]
  %a = trait.method.call %one @Tr[i32]::@value() : () -> i64 by @P_one
  %b = trait.method.call %two @Tr[i32]::@value() : () -> i64 by @P_two
  %s = arith.addi %a, %b : i64
  return %s : i64
}
