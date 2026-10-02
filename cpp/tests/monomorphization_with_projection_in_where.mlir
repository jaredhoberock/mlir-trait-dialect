// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests that a where-clause requirement containing a projection type
// (e.g., "where Self::Item : Printable") resolves correctly during
// monomorphization.  Proving @Iterable[i32] requires the pass to
// resolve the where-clause obligation @Printable[!trait.proj<@Iterable[i32], "Item">]
// into @Printable[i64] and find a matching impl.

!S = !trait.poly<0>

trait.trait private @Printable(%self: !trait.claim<@Printable[!S]>) {
  trait.method @print(!S) -> i32
}

trait.trait private @Iterable(%self: !trait.claim<@Iterable[!S]>) -> !trait.claim<@Printable[!trait.proj<@Iterable[!S], "Item">]> {
  trait.assoc_type @Item
  trait.method @first(!S) -> !trait.proj<@Iterable[!S], "Item">
}

trait.impl private @Printable_impl(%self_claim: !trait.claim<@Printable[i64]>) {
  trait.method @print(%self: i64) -> i32 {
    %c = arith.trunci %self : i64 to i32
    trait.return %c : i32
  }
}

trait.impl private @Iterable_impl(%self_claim: !trait.claim<@Iterable[i32]>) {
  trait.assoc_type @Item = i64
  trait.method @first(%self: i32) -> i64 {
    %c = arith.extsi %self : i32 to i64
    trait.return %c : i64
  }
  %req0 = trait.allege @Printable[!trait.proj<@Iterable[i32], "Item">]
  trait.return %req0 : !trait.claim<@Printable[!trait.proj<@Iterable[i32], "Item">]>
}

// The key test: proving @Iterable[i32] triggers its where-clause
// @Printable[!trait.proj<@Iterable[i32], "Item">] which must resolve
// the projection to i64 and then find impl @Printable[i64].
// CHECK-LABEL: func.func @caller
// CHECK-NOT: trait.allege
// CHECK-NOT: !trait.proj
// CHECK: call @
// CHECK-SAME: (i32) -> i64
// CHECK: return %{{.*}} : i64
func.func @caller() -> !trait.proj<@Iterable[i32], "Item"> {
  %a = trait.allege @Iterable[i32]
  %x = arith.constant 7 : i32
  %item = trait.method.call %a @Iterable[i32]::@first(%x)
    : (i32) -> !trait.proj<@Iterable[i32], "Item">
  return %item : !trait.proj<@Iterable[i32], "Item">
}
