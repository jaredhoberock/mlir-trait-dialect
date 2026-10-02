// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @g selects between its two parameters, each @Mark[!T], and the call supplies
// them through two impls of @Mark[i32]: @Seven and @Nine. The select's result
// is spelled with no proof, and no position says which one it carries, so the
// instance is refused, naming both, before any rewrite can fold the select to
// either operand.

// CHECK: error: 'arith.select' op is left with '!trait.claim<@Mark[i32]>', and this instance is supplied '!trait.claim<@Mark[i32]>' by two proofs, @Seven and @Nine; no position says which this value carries

!T = !trait.poly<0>
trait.trait private @Mark[!T] { trait.method @value() -> i64 }
trait.impl private @Seven for @Mark[i32] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Nine for @Mark[i32] {
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
