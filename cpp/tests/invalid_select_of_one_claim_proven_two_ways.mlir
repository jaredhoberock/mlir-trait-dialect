// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @g selects between its two parameters, each @Mark[!T], and the call supplies
// them through two impls of @Mark[i32]: @Seven and @Nine. A select's result is
// a join of the two values it chooses between, and a claim names one proof,
// so the instance's select, whose proof would depend on the condition, has no
// type: it is refused where it stands, naming what each value carries, and
// neither value's proof is chosen.

// CHECK: error: unproven monomorphic claim '!trait.claim<@Mark[i32]>' after instantiate-monomorphs
// CHECK: note: control flow joins it from '!trait.claim<@Mark[i32] by @Seven>' and '!trait.claim<@Mark[i32] by @Nine>'

!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.impl private @Seven(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Seven.value] {
}
func.func private @Seven.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Nine(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Nine.value] {
}
func.func private @Nine.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
func.func private @g(%a: !trait.claim<@Mark[!T]>, %b: !trait.claim<@Mark[!T]>) -> i64 {
  %c = arith.constant true
  %s = arith.select %c, %a, %b : !trait.claim<@Mark[!T]>
  %v = trait.func.call @Mark.value(%s) : (!trait.claim<@Mark[!T]>) -> i64
  return %v : i64
}
func.func @main() -> i64 {
  %a = trait.witness @Seven for @Mark[i32]
  %b = trait.witness @Nine for @Mark[i32]
  %v = trait.func.call @g(%a, %b) : (!trait.claim<@Mark[i32] by @Seven>, !trait.claim<@Mark[i32] by @Nine>) -> i64
  return %v : i64
}
