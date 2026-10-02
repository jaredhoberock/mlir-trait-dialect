// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @A_gen returns its trait's requirement @C[T] as a derive of @C_via_B over its
// where argument. Projecting the requirement off @PA, whose premise is @B_x,
// reads that derive by position and proves it by the proof whose body it is,
// @C_via_B over @B_x: the call runs @B_x's method through it and prints 1,
// although @C_i64 also implements @C[i64] and @B_y implements @B[i64].

// CHECK: {{^}}1{{$}}

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) { trait.method @b() -> i64 }
trait.trait private @C(%self: !trait.claim<@C[!T]>) { trait.method @c() -> i64 }
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@C[!T]> {}
trait.impl private @B_x(%self: !trait.claim<@B[i64]>) {
  trait.method @b() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_y(%self: !trait.claim<@B[i64]>) {
  trait.method @b() -> i64 {
    %c = arith.constant 2 : i64
    trait.return %c : i64
  }
}
trait.impl private @C_via_B(%self: !trait.claim<@C[!T]>, %b: !trait.claim<@B[!T]>) {
  trait.method @c() -> i64 {
    %v = trait.method.call %b @B[!T]::@b() : () -> i64
    trait.return %v : i64
  }
}
trait.impl private @C_i64(%self: !trait.claim<@C[i64]>) {
  trait.method @c() -> i64 {
    %c = arith.constant 100 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_gen(%self: !trait.claim<@A[!T]>, %b: !trait.claim<@B[!T]>) {
  %c = trait.derive @C[!T] from @C_via_B given(%b) : (!trait.claim<@B[!T]>)
  trait.return %c : !trait.claim<@C[!T]>
}
trait.proof private @PA {
  %b = trait.witness @B_x for @B[i64]
  %d = trait.derive @A[i64] from @A_gen given(%b) : (!trait.claim<@B[i64] by @B_x>)
  trait.return %d : !trait.claim<@A[i64]>
}
func.func @main() -> i64 {
  %a = trait.witness @PA for @A[i64]
  %c = trait.project %a[0] : !trait.claim<@A[i64] by @PA> -> !trait.claim<@C[i64]>
  %v = trait.method.call %c @C[i64]::@c() : () -> i64
  return %v : i64
}
