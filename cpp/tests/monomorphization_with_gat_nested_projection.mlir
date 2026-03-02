// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests GAT projections nested inside other GAT projections.
//
// trait Inner {
//   type Item;
// }
// trait Outer {
//   type Wrap<T>;
//   fn wrap<T>(self: Self, x: T) -> Self::Wrap<T>;
// }
// impl Inner for i32 { type Item = i64; }
// impl Outer for i1 { type Wrap<T> = T; fn wrap<T>(self, x) -> T { x } }
//
// caller: Outer[i1]::Wrap<Inner[i32]::Item>
//   = Outer[i1]::Wrap<i64>
//   = i64

!S = !trait.poly<0>
!T = !trait.poly<1>

trait.trait @Inner[!S] {
  trait.assoc_type @Item
  func.func private @get(!S) -> !trait.proj<@Inner[!S], "Item">
}

trait.trait @Outer[!S] {
  trait.assoc_type @Wrap<[!T]>
  func.func private @wrap(!S, !T) -> !trait.proj<@Outer[!S], "Wrap", [!T]>
}

trait.impl for @Inner[i32] {
  trait.assoc_type @Item = i64
  func.func @get(%self: i32) -> i64 {
    %c = arith.extsi %self : i32 to i64
    return %c : i64
  }
}

trait.impl for @Outer[i1] {
  trait.assoc_type @Wrap<[!T]> = !T
  func.func @wrap(%self: i1, %x: !T) -> !T {
    return %x : !T
  }
}

// The projection nests: Wrap<Inner[i32]::Item> -> Wrap<i64> -> i64
// CHECK-LABEL: func.func @caller() -> i64
// CHECK: call @{{[^(]*}}({{.*}}) : (i32) -> i64
// CHECK: call @{{[^(]*}}({{.*}}) : (i1, i64) -> i64
// CHECK: return %{{.*}} : i64
func.func @caller() -> !trait.proj<@Outer[i1], "Wrap", [!trait.proj<@Inner[i32], "Item">]> {
  %a = trait.allege @Inner[i32]
  %x = arith.constant 7 : i32
  %item = trait.method.call %a @Inner[i32]::@get(%x)
    : (i32) -> !trait.proj<@Inner[i32], "Item">

  %b = trait.allege @Outer[i1]
  %self = arith.constant 1 : i1
  %r = trait.method.call %b @Outer[i1]::@wrap(%self, %item)
    : (i1, !trait.proj<@Inner[i32], "Item">) -> !trait.proj<@Outer[i1], "Wrap", [!trait.proj<@Inner[i32], "Item">]>
  return %r : !trait.proj<@Outer[i1], "Wrap", [!trait.proj<@Inner[i32], "Item">]>
}
