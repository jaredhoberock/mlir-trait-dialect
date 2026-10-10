// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Two proofs of A[i64] derive it from one impl with different premises:
// @by_one cites @B_one and @by_ten cites @B_ten. A proof is identified by its
// body, so the method calls through the two proofs reach two instances of
// @A_gen's method, each running the impl its proof cited: 1 + 10.

// CHECK: {{^}}11{{$}}

!P = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!P]>) { trait.method @b() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!P]>) { trait.method @a() -> i64 }
trait.impl private @B_one(%self: !trait.claim<@B[i64]>) {
  trait.method @b() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_ten(%self: !trait.claim<@B[i64]>) {
  trait.method @b() -> i64 {
    %c = arith.constant 10 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_gen(%self: !trait.claim<@A[!P]>, %b: !trait.claim<@B[!P]>) {
  trait.method @a() -> i64 {
    %r = trait.method.call %b @B[!P]::@b() : () -> i64
    trait.return %r : i64
  }
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
  %x = trait.method.call %one @A[i64]::@a() : () -> i64 by @by_one
  %y = trait.method.call %ten @A[i64]::@a() : () -> i64 by @by_ten
  %s = arith.addi %x, %y : i64
  return %s : i64
}
