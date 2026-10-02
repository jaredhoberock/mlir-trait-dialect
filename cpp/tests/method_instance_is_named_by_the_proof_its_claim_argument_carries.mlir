// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @Tr's method takes a claim of its own, and two calls through one receiver
// supply it for different reasons: @Mark_one (7) and @Mark_two (9). A method
// instance is named by everything the call supplies at each of its positions,
// so the calls reach two instances of @Tr_i32's method, each running the impl
// its own claim argument selects.

// CHECK: {{^}}16{{$}}

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.method @run(!trait.claim<@Mark[!T]>) -> i64 }
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
trait.impl private @Tr_i32(%self: !trait.claim<@Tr[i32]>) {
  trait.method @run(%m: !trait.claim<@Mark[i32]>) -> i64 {
    %v = trait.method.call %m @Mark[i32]::@value() : () -> i64
    trait.return %v : i64
  }
}
func.func @main() -> i64 {
  %tr = trait.witness @Tr_i32 for @Tr[i32]
  %one = trait.witness @Mark_one for @Mark[i32]
  %two = trait.witness @Mark_two for @Mark[i32]
  %a = trait.method.call %tr @Tr[i32]::@run(%one) : (!trait.claim<@Mark[i32] by @Mark_one>) -> i64 by @Tr_i32
  %b = trait.method.call %tr @Tr[i32]::@run(%two) : (!trait.claim<@Mark[i32] by @Mark_two>) -> i64 by @Tr_i32
  %s = arith.addi %a, %b : i64
  return %s : i64
}
