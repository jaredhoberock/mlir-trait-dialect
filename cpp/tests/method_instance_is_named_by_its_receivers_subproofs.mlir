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
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @value() -> i64 }
trait.impl private @Mark_one(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Mark_two(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.impl private @Tr_impl(%self: !trait.claim<@Tr[i32]>, %mark: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = trait.method.call %mark @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
trait.proof private @P_one {
  %p0 = trait.witness @Mark_one for @Mark[i32]
  %d = trait.derive @Tr[i32] from @Tr_impl given(%p0) : (!trait.claim<@Mark[i32] by @Mark_one>)
  trait.return %d : !trait.claim<@Tr[i32]>
}
trait.proof private @P_two {
  %p0 = trait.witness @Mark_two for @Mark[i32]
  %d = trait.derive @Tr[i32] from @Tr_impl given(%p0) : (!trait.claim<@Mark[i32] by @Mark_two>)
  trait.return %d : !trait.claim<@Tr[i32]>
}
func.func @main() -> i64 {
  %one = trait.witness @P_one for @Tr[i32]
  %two = trait.witness @P_two for @Tr[i32]
  %a = trait.method.call %one @Tr[i32]::@value() : () -> i64 by @P_one
  %b = trait.method.call %two @Tr[i32]::@value() : () -> i64 by @P_two
  %s = arith.addi %a, %b : i64
  return %s : i64
}
