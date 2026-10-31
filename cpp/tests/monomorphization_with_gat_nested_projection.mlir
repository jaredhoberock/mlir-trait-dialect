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

trait.trait private @Inner(!trait.claim<@Inner[!S]>) assoc {Item = 0} members [@Inner.get]
func.func private @Inner.get(!trait.claim<@Inner[!S]>, !S) -> !trait.proj<@Inner[!S], "Item">

trait.trait private @Outer(!trait.claim<@Outer[!S]>) assoc {Wrap = 1} members [@Outer.wrap]
func.func private @Outer.wrap(!trait.claim<@Outer[!S]>, !S, !T) -> !trait.proj<@Outer[!S], "Wrap", [!T]>

trait.impl private @Inner_impl(%self_claim: !trait.claim<@Inner[i32]>) bindings {Item = i64} members [@Inner.get -> @Inner_impl.get] {
}
func.func private @Inner_impl.get(%self_claim: !trait.claim<@Inner[i32]>, %self: i32) -> i64 {
  %c = arith.extsi %self : i32 to i64
  return %c : i64
}

trait.impl private @Outer_impl(%self_claim: !trait.claim<@Outer[i1]>) bindings {Wrap = !trait.poly<0>} members [@Outer.wrap -> @Outer_impl.wrap] {
}
func.func private @Outer_impl.wrap(%self_claim: !trait.claim<@Outer[i1]>, %self: i1, %x: !trait.poly<0>) -> !trait.poly<0> {
  return %x : !trait.poly<0>
}

// The projection nests: Wrap<Inner[i32]::Item> -> Wrap<i64> -> i64
// CHECK-LABEL: func.func @caller() -> i64
// CHECK: call @{{[^(]*}}({{.*}}) : (i32) -> i64
// CHECK: call @{{[^(]*}}({{.*}}) : (i1, i64) -> i64
// CHECK: return %{{.*}} : i64
func.func @caller() -> !trait.proj<@Outer[i1], "Wrap", [!trait.proj<@Inner[i32], "Item">]> {
  %a = trait.allege @Inner[i32]
  %x = arith.constant 7 : i32
  %item = trait.func.call @Inner.get(%a, %x) : (!trait.claim<@Inner[i32]>, i32) -> !trait.proj<@Inner[i32], "Item">

  %b = trait.allege @Outer[i1]
  %self = arith.constant 1 : i1
  %r = trait.func.call @Outer.wrap(%b, %self, %item) : (!trait.claim<@Outer[i1]>, i1, !trait.proj<@Inner[i32], "Item">) -> !trait.proj<@Outer[i1], "Wrap", [!trait.proj<@Inner[i32], "Item">]>
  return %r : !trait.proj<@Outer[i1], "Wrap", [!trait.proj<@Inner[i32], "Item">]>
}
