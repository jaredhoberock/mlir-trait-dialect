// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file 2>&1 | FileCheck %s

// The impl verifier reads an impl method's signature, and the trait's, through
// the impl's own bindings, under plain verification. @I binds Item<U> beneath
// a tuple to its own projection at a larger argument, once and then twice, and
// the third @I's method takes a plain type where the trait's spells the
// projection: what a binding spells beneath its root is read nested in the
// projection's reading, bounded by the depth limit, so each reading stops
// there and the impl is refused rather than recursing until the stack is gone.

// CHECK: :[[# @LINE + 8]]:1: error: 'trait.impl' op projection normalization did not converge
// CHECK: :[[# @LINE + 21]]:1: error: 'trait.impl' op projection normalization did not converge
// CHECK: :[[# @LINE + 34]]:1: error: 'trait.impl' op projection normalization did not converge

trait.trait private @A(%s: !trait.claim<@A[!trait.poly<0>]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
  trait.method @m(!trait.proj<@A[!trait.poly<0>], "Item", [i64]>) -> i64
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>
  trait.method @m(%x: !trait.proj<@A[i32], "Item", [i64]>) -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

// -----

trait.trait private @A(%s: !trait.claim<@A[!trait.poly<0>]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
  trait.method @m(!trait.proj<@A[!trait.poly<0>], "Item", [i64]>) -> i64
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>
  trait.method @m(%x: !trait.proj<@A[i32], "Item", [i64]>) -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}

// -----

trait.trait private @A(%s: !trait.claim<@A[!trait.poly<0>]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
  trait.method @m(!trait.proj<@A[!trait.poly<0>], "Item", [i64]>) -> ()
}
trait.impl private @I(%s: !trait.claim<@A[i32]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>
  trait.method @m(%x: i64) -> () { trait.return }
}
