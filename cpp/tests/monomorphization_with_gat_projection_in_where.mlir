// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests that a where-clause requirement containing a GAT projection
// resolves correctly during monomorphization.
//
// trait Printable { fn print(self: Self) -> i32; }
// trait Transform {
//   type Output<T>;
//   fn apply<T>(self: Self, x: T) -> Self::Output<T>;
// }
// where Self::Output<i32> : Printable
//
// Proving @Transform[i64] requires resolving the where-clause obligation
// @Printable[!trait.proj<@Transform[i64], "Output", [i32]>].
// With impl Transform for i64 { type Output<T> = T }, the projection
// resolves to i32, and @Printable[i32] is satisfied by impl Printable for i32.

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait private @Printable(%self: !trait.claim<@Printable[!S]>) {
  trait.method @print(!S) -> i32
}

trait.trait private @Transform(%self: !trait.claim<@Transform[!S]>) -> !trait.claim<@Printable[!trait.proj<@Transform[!S], "Output", [i32]>]> {
  trait.assoc_type @Output<[!T]>
  trait.method @apply(!S, !T) -> !trait.proj<@Transform[!S], "Output", [!T]>
}

trait.impl private @Printable_impl(%self_claim: !trait.claim<@Printable[i32]>) {
  trait.method @print(%self: i32) -> i32 {
    trait.return %self : i32
  }
}

trait.impl private @Transform_impl(%self_claim: !trait.claim<@Transform[i64]>) {
  trait.assoc_type @Output<[!trait.poly<0>]> = !trait.poly<0>
  trait.method @apply(%self: i64, %x: !trait.poly<0>) -> !trait.poly<0> {
    trait.return %x : !trait.poly<0>
  }
  %req0 = trait.allege @Printable[!trait.proj<@Transform[i64], "Output", [i32]>]
  trait.return %req0 : !trait.claim<@Printable[!trait.proj<@Transform[i64], "Output", [i32]>]>
}

// Proving @Transform[i64] must resolve the where clause:
//   @Printable[!trait.proj<@Transform[i64], "Output", [i32]>]
//   -> @Printable[i32]  (via GAT projection resolution)
//   -> satisfied by impl @Printable[i32]
// CHECK-LABEL: func.func @caller
// CHECK-NOT: trait.allege
// CHECK: call @{{[^(]*}}({{.*}}) : (i64, i1) -> i1
// CHECK: return %{{.*}} : i1
func.func @caller() -> !trait.proj<@Transform[i64], "Output", [i1]> {
  %a = trait.allege @Transform[i64]
  %self = arith.constant 7 : i64
  %x = arith.constant 1 : i1
  %r = trait.method.call %a @Transform[i64]::@apply(%self, %x)
    : (i64, i1) -> !trait.proj<@Transform[i64], "Output", [i1]>
  return %r : !trait.proj<@Transform[i64], "Output", [i1]>
}
