// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A method's claim parameters are the hypotheses its body is judged under, as
// an isolated function's are, although a method is not isolated from above:
// @use's equality parameter says @Trait[!U]'s Item is i64, which is what lets
// the call read @inner's result as i64.

// CHECK-LABEL: trait.method @use
// CHECK: trait.func.call @inner

!T = !trait.poly<0>
trait.trait private @Trait[!T] {
  trait.assoc_type @Item
}

!A = !trait.poly<2>
func.func private @inner(!A, !trait.claim<@Trait[!A]>) -> !trait.proj<@Trait[!A], "Item">

!U = !trait.poly<1>
trait.trait private @User[!U] {
  trait.method @use(%x: !U, %t: !trait.claim<@Trait[!U]>,
      %eq: !trait.claim<!trait.proj<@Trait[!U], "Item"> = i64>) -> i64 {
    %r = trait.func.call @inner(%x, %t) : (!U, !trait.claim<@Trait[!U]>) -> i64
    trait.return %r : i64
  }
}
