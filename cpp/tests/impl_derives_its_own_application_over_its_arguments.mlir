// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// An impl's evidence of its own application is a derive of the impl over its
// own block arguments: standing in the impl block and read by @twice, or
// inside the member that reads it, @thrice. Both are cut at A[i32]:
// 3 + 3 and 3 + 3 + 3.

// CHECK: {{^}}15{{$}}

!T = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!T]>) members [@B.b]
func.func private @B.b(!trait.claim<@B[!T]>, !T) -> i64
trait.trait private @A(!trait.claim<@A[!T]>) members [@A.a, @A.twice, @A.thrice]
func.func private @A.a(!trait.claim<@A[!T]>, !T) -> i64
func.func private @A.twice(!trait.claim<@A[!T]>, !T) -> i64
func.func private @A.thrice(!trait.claim<@A[!T]>, !T) -> i64
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %c = arith.constant 3 : i64
  return %c : i64
}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>) members [@A.a -> @A_gen.a, @A.twice -> @A_gen.twice, @A.thrice -> @A_gen.thrice] {
}
func.func private @A_gen.a(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %x: !T) -> i64 {
  %v = trait.func.call @B.b(%b, %x) : (!trait.claim<@B[!T]>, !T) -> i64
  return %v : i64
}
func.func private @A_gen.twice(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %x: !T) -> i64 {
  %own = trait.derive @A[!T] from @A_gen[!trait.poly<0>] given(%b) : (!trait.claim<@B[!T]>)
  %u = trait.func.call @A.a(%own, %x) : (!trait.claim<@A[!T]>, !T) -> i64
  %r = arith.addi %u, %u : i64
  return %r : i64
}
func.func private @A_gen.thrice(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>, %x: !T) -> i64 {
  %me = trait.derive @A[!T] from @A_gen[!trait.poly<0>] given(%b) : (!trait.claim<@B[!T]>)
  %u = trait.func.call @A.a(%me, %x) : (!trait.claim<@A[!T]>, !T) -> i64
  %v = arith.addi %u, %u : i64
  %r = arith.addi %v, %u : i64
  return %r : i64
}
func.func @main() -> i64 {
  %x = arith.constant 0 : i32
  %a = trait.allege @A[i32]
  %r = trait.func.call @A.twice(%a, %x) : (!trait.claim<@A[i32]>, i32) -> i64
  %s = trait.func.call @A.thrice(%a, %x) : (!trait.claim<@A[i32]>, i32) -> i64
  %t = arith.addi %r, %s : i64
  return %t : i64
}
