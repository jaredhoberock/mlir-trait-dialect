// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A trait default member reads its trait's own claim, the trait's block
// argument, and selects the trait's requirement off it. The default is cut
// straight from the trait at A[i32]: @a through the claim and @b through the
// projected requirement, 4 + 3.

// CHECK: {{^}}7{{$}}

!T = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!T]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!T]>, !T) -> i64
trait.trait private @A(!trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> members [@A.a, @A.both]
func.func private @A.a(!trait.claim<@A[!T]>, !T) -> i64
func.func private @A.both(%self: !trait.claim<@A[!T]>, %x: !T) -> i64 {
  %b = trait.project %self[0] : !trait.claim<@A[!T]> -> !trait.claim<@B[!T]>
  %u = trait.func.call @A.a(%self, %x) : (!trait.claim<@A[!T]>, !T) -> i64
  %v = trait.func.call @B.b(%b, %x) : (!trait.claim<@B[!T]>, !T) -> i64
  %r = arith.addi %u, %v : i64
  return %r : i64
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %c = arith.constant 3 : i64
  return %c : i64
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
  %b = trait.witness @B_i32 for @B[i32]
  trait.return %b : !trait.claim<@B[i32] by @B_i32>
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>, %x: i32) -> i64 {
  %c = arith.constant 4 : i64
  return %c : i64
}
func.func @main() -> i64 {
  %x = arith.constant 0 : i32
  %a = trait.allege @A[i32]
  %r = trait.func.call @A.both(%a, %x) : (!trait.claim<@A[i32]>, i32) -> i64
  return %r : i64
}
