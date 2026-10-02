// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// An impl's evidence of its own application is a derive of the impl over its
// own block arguments: standing in the impl block and read by @twice, or
// inside the method that reads it, @thrice. Both are cut at A[i32]:
// 3 + 3 and 3 + 3 + 3.

// CHECK: {{^}}15{{$}}

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {
  trait.method @b(!T) -> i64
}
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
  trait.method @a(!T) -> i64
  trait.method @twice(!T) -> i64
  trait.method @thrice(!T) -> i64
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @b(%x: i32) -> i64 {
    %c = arith.constant 3 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>) {
  %own = trait.derive @A[!T] from @A_gen given(%b) : (!trait.claim<@B[!T]>)
  trait.method @a(%x: !T) -> i64 {
    %v = trait.method.call %b @B[!T]::@b(%x) : (!T) -> i64
    trait.return %v : i64
  }
  trait.method @twice(%x: !T) -> i64 {
    %u = trait.method.call %own @A[!T]::@a(%x) : (!T) -> i64
    %r = arith.addi %u, %u : i64
    trait.return %r : i64
  }
  trait.method @thrice(%x: !T) -> i64 {
    %me = trait.derive @A[!T] from @A_gen given(%b) : (!trait.claim<@B[!T]>)
    %u = trait.method.call %me @A[!T]::@a(%x) : (!T) -> i64
    %v = arith.addi %u, %u : i64
    %r = arith.addi %v, %u : i64
    trait.return %r : i64
  }
}
func.func @main() -> i64 {
  %x = arith.constant 0 : i32
  %a = trait.allege @A[i32]
  %r = trait.method.call %a @A[i32]::@twice(%x) : (i32) -> i64
  %s = trait.method.call %a @A[i32]::@thrice(%x) : (i32) -> i64
  %t = arith.addi %r, %s : i64
  return %t : i64
}
