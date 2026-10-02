// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A trait default method reads its trait's own claim, the trait's block
// argument, and selects the trait's requirement off it. The default is cut
// straight from the trait at A[i32]: @a through the claim and @b through the
// projected requirement, 4 + 3.

// CHECK: {{^}}7{{$}}

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {
  trait.method @b(!T) -> i64
}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {
  trait.method @a(!T) -> i64
  trait.method @both(%x: !T) -> i64 {
    %b = trait.project %self[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
    %u = trait.method.call %self @A[!T]::@a(%x) : (!T) -> i64
    %v = trait.method.call %b @B[!T]::@b(%x) : (!T) -> i64
    %r = arith.addi %u, %v : i64
    trait.return %r : i64
  }
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @b(%x: i32) -> i64 {
    %c = arith.constant 3 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @a(%x: i32) -> i64 {
    %c = arith.constant 4 : i64
    trait.return %c : i64
  }
  %b = trait.witness @B_i32 for @B[i32]
  trait.return %b : !trait.claim<@B[i32] by @B_i32>
}
func.func @main() -> i64 {
  %x = arith.constant 0 : i32
  %a = trait.allege @A[i32]
  %r = trait.method.call %a @A[i32]::@both(%x) : (i32) -> i64
  return %r : i64
}
