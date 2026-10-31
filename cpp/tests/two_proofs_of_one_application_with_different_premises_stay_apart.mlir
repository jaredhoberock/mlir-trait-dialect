// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Two proofs of A[i64] derive it from one impl with different premises:
// @by_one cites @B_one and @by_ten cites @B_ten. A proof is identified by its
// body, so the method calls through the two proofs reach two instances of
// @A_gen's method, each running the impl its proof cited: 1 + 10.

// CHECK: {{^}}11{{$}}

!P = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!P]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!P]>) -> i64
trait.trait private @A(!trait.claim<@A[!P]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!P]>) -> i64
trait.impl private @B_one(%self: !trait.claim<@B[i64]>) members [@B.b -> @B_one.b] {
}
func.func private @B_one.b(%self: !trait.claim<@B[i64]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
trait.impl private @B_ten(%self: !trait.claim<@B[i64]>) members [@B.b -> @B_ten.b] {
}
func.func private @B_ten.b(%self: !trait.claim<@B[i64]>) -> i64 {
  %c = arith.constant 10 : i64
  return %c : i64
}
trait.impl private @A_gen(%self: !trait.claim<@A[!P]>, %b: !trait.claim<@B[!P]>) members [@A.a -> @A_gen.a] {
}
func.func private @A_gen.a(%self: !trait.claim<@A[!P]>, %b: !trait.claim<@B[!P]>) -> i64 {
  %r = trait.func.call @B.b(%b) : (!trait.claim<@B[!P]>) -> i64
  return %r : i64
}
trait.proof private @by_one {
  %b = trait.witness @B_one for @B[i64]
  %d = trait.derive @A[i64] from @A_gen[i64] given(%b) : (!trait.claim<@B[i64] by @B_one>)
  trait.return %d : !trait.claim<@A[i64]>
}
trait.proof private @by_ten {
  %b = trait.witness @B_ten for @B[i64]
  %d = trait.derive @A[i64] from @A_gen[i64] given(%b) : (!trait.claim<@B[i64] by @B_ten>)
  trait.return %d : !trait.claim<@A[i64]>
}
func.func @main() -> i64 {
  %one = trait.witness @by_one for @A[i64]
  %ten = trait.witness @by_ten for @A[i64]
  %x = trait.func.call @A.a(%one) : (!trait.claim<@A[i64] by @by_one>) -> i64
  %y = trait.func.call @A.a(%ten) : (!trait.claim<@A[i64] by @by_ten>) -> i64
  %s = arith.addi %x, %y : i64
  return %s : i64
}
