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

trait.trait private @Printable(!trait.claim<@Printable[!S]>) members [@Printable.print]
func.func private @Printable.print(!trait.claim<@Printable[!S]>, !S) -> i32

trait.trait private @Transform(!trait.claim<@Transform[!S]>) -> !trait.claim<@Printable[!trait.proj<@Transform[!S], "Output", [i32]>]> assoc {Output = 1} members [@Transform.apply]
func.func private @Transform.apply(!trait.claim<@Transform[!S]>, !S, !T) -> !trait.proj<@Transform[!S], "Output", [!T]>

trait.impl private @Printable_impl(%self_claim: !trait.claim<@Printable[i32]>) members [@Printable.print -> @Printable_impl.print] {
}
func.func private @Printable_impl.print(%self_claim: !trait.claim<@Printable[i32]>, %self: i32) -> i32 {
  return %self : i32
}

trait.impl private @Transform_impl(%self_claim: !trait.claim<@Transform[i64]>) bindings {Output = !trait.poly<0>} members [@Transform.apply -> @Transform_impl.apply] {
  %req0 = trait.allege @Printable[!trait.proj<@Transform[i64], "Output", [i32]>]
  trait.return %req0 : !trait.claim<@Printable[!trait.proj<@Transform[i64], "Output", [i32]>]>
}
func.func private @Transform_impl.apply(%self_claim: !trait.claim<@Transform[i64]>, %self: i64, %x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
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
  %r = trait.func.call @Transform.apply(%a, %self, %x) : (!trait.claim<@Transform[i64]>, i64, i1) -> !trait.proj<@Transform[i64], "Output", [i1]>
  return %r : !trait.proj<@Transform[i64], "Output", [i1]>
}
