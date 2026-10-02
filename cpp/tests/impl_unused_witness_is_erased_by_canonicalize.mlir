// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -canonicalize | FileCheck %s

// An impl's block holds evidence values beside its methods. Canonicalization
// erases a witness nothing reads, and the impl still returns the one its
// trait's requirement reads.

// CHECK-LABEL: trait.impl private @A_i32(%self: !trait.claim<@A[i32]>)
// CHECK-NEXT:    %[[B:.*]] = trait.witness @B_i32 for @B[i32]
// CHECK-NEXT:    trait.method @a
// CHECK:         trait.return %[[B]] : !trait.claim<@B[i32] by @B_i32>
// CHECK-NOT:     trait.witness

!T = !trait.poly<0>
trait.trait private @B(%self: !trait.claim<@B[!T]>) {}
trait.trait private @A(%self: !trait.claim<@A[!T]>) -> !trait.claim<@B[!T]> {
  trait.method @a(!T) -> i64
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  %unused = trait.witness @B_i32 for @B[i32]
  %b = trait.witness @B_i32 for @B[i32]
  trait.method @a(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    trait.return %c : i64
  }
  trait.return %b : !trait.claim<@B[i32] by @B_i32>
}
