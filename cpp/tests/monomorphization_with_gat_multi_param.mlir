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

trait.trait private @BiMap(!trait.claim<@BiMap[!S]>) assoc {Left = 2, Right = 2} members [@BiMap.left, @BiMap.right]
func.func private @BiMap.left(!trait.claim<@BiMap[!S]>, !S, !T, !U) -> !trait.proj<@BiMap[!S], "Left", [!T, !U]>
func.func private @BiMap.right(!trait.claim<@BiMap[!S]>, !S, !T, !U) -> !trait.proj<@BiMap[!S], "Right", [!T, !U]>

trait.impl private @BiMap_impl(%self_claim: !trait.claim<@BiMap[i1]>) bindings {Left = !trait.poly<0>, Right = !trait.poly<1>} members [@BiMap.left -> @BiMap_impl.left, @BiMap.right -> @BiMap_impl.right] {
}
func.func private @BiMap_impl.left(%self_claim: !trait.claim<@BiMap[i1]>, %self: i1, %a: !trait.poly<0>, %b: !trait.poly<1>) -> !trait.poly<0> {
  return %a : !trait.poly<0>
}
func.func private @BiMap_impl.right(%self_claim: !trait.claim<@BiMap[i1]>, %self: i1, %a: !trait.poly<0>, %b: !trait.poly<1>) -> !trait.poly<1> {
  return %b : !trait.poly<1>
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
  %r = trait.func.call @BiMap.left(%a, %self, %x, %y) : (!trait.claim<@BiMap[i1]>, i1, i32, i64) -> !trait.proj<@BiMap[i1], "Left", [i32, i64]>
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
  %r = trait.func.call @BiMap.right(%a, %self, %x, %y) : (!trait.claim<@BiMap[i1]>, i1, i32, i64) -> !trait.proj<@BiMap[i1], "Right", [i32, i64]>
  return %r : !trait.proj<@BiMap[i1], "Right", [i32, i64]>
}
