// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(resolve-impls-trait)' | FileCheck %s

// A proof is identified by the evidence its derive cites. @A_i32_p derives
// @A[i32] from @A_i32 over @b_custom, while selection proves the premise
// @B[i32] by @B_i32 itself, so the allegation is answered by a proof over the
// premises selection chose, written beside @A_i32_p rather than read from it.

// CHECK: trait.proof private @A_i32_p {
// CHECK:   trait.witness @b_custom for @B[i32]
// CHECK: func.func @main
// CHECK:   trait.witness @[[P:A_i32_p_h[0-9a-f]+]] for @A[i32]
// CHECK: trait.proof private @[[P]] {
// CHECK:   trait.witness @B_i32 for @B[i32]
// CHECK:   trait.derive @A[i32] from @A_i32

trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) { trait.method @v() -> i64 }
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @v() -> i64 {
    %c = arith.constant 2 : i64
    trait.return %c : i64
  }
}
trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>, %b: !trait.claim<@B[i32]>) {
  trait.method @a() -> i64 {
    %r = trait.method.call %b @B[i32]::@v() : () -> i64
    trait.return %r : i64
  }
}
trait.proof private @b_custom {
  %d = trait.derive @B[i32] from @B_i32 given()
  trait.return %d : !trait.claim<@B[i32]>
}
trait.proof private @A_i32_p {
  %b = trait.witness @b_custom for @B[i32]
  %d = trait.derive @A[i32] from @A_i32 given(%b) : (!trait.claim<@B[i32] by @b_custom>)
  trait.return %d : !trait.claim<@A[i32]>
}
func.func @main() -> i64 {
  %e = trait.allege @A[i32]
  %r = trait.method.call %e @A[i32]::@a() : () -> i64
  return %r : i64
}
