// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -canonicalize | FileCheck %s

// An impl's block contains the evidence values its return reads. Canonicalization
// erases a witness nothing reads, and the impl still returns the one its
// trait's requirement reads; its member function stands beside it.

// CHECK-LABEL: trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a]
// CHECK-NEXT:    %[[B:.*]] = trait.witness @B_i32 for @B[i32]
// CHECK-NEXT:    trait.return %[[B]] : !trait.claim<@B[i32] by @B_i32>
// CHECK-NOT:     trait.witness
// CHECK:         func.func private @A_i32.a

!T = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!T]>)
trait.trait private @A(!trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> members [@A.a]
func.func private @A.a(!trait.claim<@A[!T]>, !T) -> i64
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
  %unused = trait.witness @B_i32 for @B[i32]
  %b = trait.witness @B_i32 for @B[i32]
  trait.return %b : !trait.claim<@B[i32] by @B_i32>
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>, %x: i32) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}
