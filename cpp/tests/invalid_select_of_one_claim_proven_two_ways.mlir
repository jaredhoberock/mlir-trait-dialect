// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @g selects between its two parameters, each @Mark[!T], and the call supplies
// them through two impls of @Mark[i32]: @Seven and @Nine. A claim's proof is
// part of its type, so the instance's two arms are of two types, and the
// select, whose arms and result share one type, refuses the instance: which
// impl the select's result runs is decided by no position.

// CHECK: error: 'arith.select' op failed to verify that all of {true_value, false_value, result} have same type

!T = !trait.poly<0>
trait.trait private @Mark(%self: !trait.claim<@Mark[!T]>) { trait.method @value() -> i64 }
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
func.func private @g(%a: !trait.claim<@Mark[!T]>, %b: !trait.claim<@Mark[!T]>) -> i64 {
  %c = arith.constant true
  %s = arith.select %c, %a, %b : !trait.claim<@Mark[!T]>
  %v = trait.method.call %s @Mark[!T]::@value() : () -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %a = trait.witness @Seven for @Mark[i32]
  %b = trait.witness @Nine for @Mark[i32]
  %v = trait.func.call @g(%a, %b) : (!trait.claim<@Mark[i32] by @Seven>, !trait.claim<@Mark[i32] by @Nine>) -> i64
  return %v : i64
}
