// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests GAT with multiple type parameters.
//
// trait BiMap {
//   type Left<T, U>;
//   type Right<T, U>;
//   fn left<T, U>(self: Self, a: T, b: U) -> Self::Left<T, U>;
//   fn right<T, U>(self: Self, a: T, b: U) -> Self::Right<T, U>;
// }
// impl BiMap for i1 {
//   type Left<T, U> = T;
//   type Right<T, U> = U;
//   fn left<T, U>(self: i1, a: T, b: U) -> T { a }
//   fn right<T, U>(self: i1, a: T, b: U) -> U { b }
// }

!S = !trait.poly<0>
!T = !trait.poly<1>
!U = !trait.poly<2>

trait.trait @BiMap[!S] {
  trait.assoc_type @Left<[!T, !U]>
  trait.assoc_type @Right<[!T, !U]>
  func.func private @left(!S, !T, !U) -> !trait.proj<@BiMap[!S], "Left", [!T, !U]>
  func.func private @right(!S, !T, !U) -> !trait.proj<@BiMap[!S], "Right", [!T, !U]>
}

trait.impl for @BiMap[i1] {
  trait.assoc_type @Left<[!T, !U]> = !T
  trait.assoc_type @Right<[!T, !U]> = !U
  func.func @left(%self: i1, %a: !T, %b: !U) -> !T {
    return %a : !T
  }
  func.func @right(%self: i1, %a: !T, %b: !U) -> !U {
    return %b : !U
  }
}

// CHECK-LABEL: func.func @call_left
// CHECK-NOT: !trait.proj
// CHECK: call @
// CHECK-SAME: (i1, i32, i64) -> i32
// CHECK: return %{{.*}} : i32
func.func @call_left() -> !trait.proj<@BiMap[i1], "Left", [i32, i64]> {
  %a = trait.allege @BiMap[i1]
  %self = arith.constant 1 : i1
  %x = arith.constant 42 : i32
  %y = arith.constant 7 : i64
  %r = trait.method.call %a @BiMap[i1]::@left(%self, %x, %y)
    : (i1, i32, i64) -> !trait.proj<@BiMap[i1], "Left", [i32, i64]>
  return %r : !trait.proj<@BiMap[i1], "Left", [i32, i64]>
}

// CHECK-LABEL: func.func @call_right
// CHECK-NOT: !trait.proj
// CHECK: call @
// CHECK-SAME: (i1, i32, i64) -> i64
// CHECK: return %{{.*}} : i64
func.func @call_right() -> !trait.proj<@BiMap[i1], "Right", [i32, i64]> {
  %a = trait.allege @BiMap[i1]
  %self = arith.constant 1 : i1
  %x = arith.constant 42 : i32
  %y = arith.constant 7 : i64
  %r = trait.method.call %a @BiMap[i1]::@right(%self, %x, %y)
    : (i1, i32, i64) -> !trait.proj<@BiMap[i1], "Right", [i32, i64]>
  return %r : !trait.proj<@BiMap[i1], "Right", [i32, i64]>
}
