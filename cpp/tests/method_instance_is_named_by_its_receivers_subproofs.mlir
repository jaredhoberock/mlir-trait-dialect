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
trait.trait private @Mark(!trait.claim<@Mark[!T]>) members [@Mark.value]
func.func private @Mark.value(!trait.claim<@Mark[!T]>) -> i64
trait.trait private @Tr(!trait.claim<@Tr[!T]>) members [@Tr.value]
func.func private @Tr.value(!trait.claim<@Tr[!T]>) -> i64
trait.impl private @Mark_one(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Mark_one.value] {
}
func.func private @Mark_one.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @Mark_two(%self: !trait.claim<@Mark[i32]>) members [@Mark.value -> @Mark_two.value] {
}
func.func private @Mark_two.value(%self: !trait.claim<@Mark[i32]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}
trait.impl private @Tr_impl(%self: !trait.claim<@Tr[i32]>, %mark: !trait.claim<@Mark[i32]>) members [@Tr.value -> @Tr_impl.value] {
}
func.func private @Tr_impl.value(%self: !trait.claim<@Tr[i32]>, %mark: !trait.claim<@Mark[i32]>) -> i64 {
  %v = trait.func.call @Mark.value(%mark) : (!trait.claim<@Mark[i32]>) -> i64
  return %v : i64
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
  %a = trait.func.call @Tr.value(%one) : (!trait.claim<@Tr[i32] by @P_one>) -> i64
  %b = trait.func.call @Tr.value(%two) : (!trait.claim<@Tr[i32] by @P_two>) -> i64
  %s = arith.addi %a, %b : i64
  return %s : i64
}
