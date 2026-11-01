// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Tests forwarding associated types through monomorphization.
//
// trait Inner {
//   type Assoc;
//   fn method(self: Self) -> Self::Assoc;
// }
// impl Inner for i32 { type Assoc = f32; fn method(self) -> f32 { ... } }
//
// trait Outer {
//   type Assoc;
//   fn method(self: Self) -> Self::Assoc;
// }
// impl<U: Inner> Outer for tuple<U> {
//   type Assoc = Inner[U]::Assoc;   // forwarding!
//   fn method(self: tuple<U>) -> Inner[U]::Assoc { ... }
// }
//
// The forwarding impl's member signature, after substituting Self=tuple<U>,
// has return type Outer[tuple<U>]::Assoc. The verifier must resolve this
// to Inner[U]::Assoc (via the impl's own binding) even though tuple<U>
// is polymorphic.

// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.proof
// CHECK-NOT: trait.allege
// CHECK-NOT: trait.assume

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Inner(!trait.claim<@Inner[!S]>) assoc {Assoc = 0} members [@Inner.method]
func.func private @Inner.method(!trait.claim<@Inner[!S]>, !S) -> !trait.proj<@Inner[!S], "Assoc">

trait.trait private @Outer(!trait.claim<@Outer[!S]>) assoc {Assoc = 0} members [@Outer.method]
func.func private @Outer.method(!trait.claim<@Outer[!S]>, !S) -> !trait.proj<@Outer[!S], "Assoc">

// Concrete impl: Inner for i32, Assoc = f32
trait.impl private @Inner_i32(%self_claim: !trait.claim<@Inner[i32]>) bindings {Assoc = f32} members [@Inner.method -> @Inner_i32.method] {
}
func.func private @Inner_i32.method(%self_claim: !trait.claim<@Inner[i32]>, %self: i32) -> f32 {
  %c = arith.sitofp %self : i32 to f32
  return %c : f32
}

// Forwarding impl: Outer for tuple<U> where Inner[U], Assoc = Inner[U]::Assoc
trait.impl private @Outer_tuple(%self_claim: !trait.claim<@Outer[tuple<!trait.poly<0>>]>, %inner: !trait.claim<@Inner[!trait.poly<0>]>) bindings {Assoc = !trait.proj<@Inner[!trait.poly<0>], "Assoc">} members [@Outer.method -> @Outer_tuple.method] {
}
func.func private @Outer_tuple.method(%self_claim: !trait.claim<@Outer[tuple<!trait.poly<0>>]>, %inner: !trait.claim<@Inner[!trait.poly<0>]>, %self: tuple<!trait.poly<0>>) -> !trait.proj<@Inner[!trait.poly<0>], "Assoc"> {
  %elem = "test.extract"(%self) : (tuple<!trait.poly<0>>) -> !trait.poly<0>
  %res = trait.func.call @Inner.method(%inner, %elem) : (!trait.claim<@Inner[!trait.poly<0>]>, !trait.poly<0>) -> !trait.proj<@Inner[!trait.poly<0>], "Assoc">
  return %res : !trait.proj<@Inner[!trait.poly<0>], "Assoc">
}

// Concrete call site: call Outer::method on tuple<i32>
// Outer[tuple<i32>]::Assoc -> Inner[i32]::Assoc -> f32
// CHECK-LABEL: func.func @test_forwarding
// CHECK: call @{{[^(]*}}({{.*}}) : (tuple<i32>) -> f32
// CHECK: return %{{.*}} : f32
func.func @test_forwarding(%arg: tuple<i32>) -> !trait.proj<@Outer[tuple<i32>], "Assoc"> {
  %a = trait.allege @Outer[tuple<i32>]
  %res = trait.func.call @Outer.method(%a, %arg) : (!trait.claim<@Outer[tuple<i32>]>, tuple<i32>) -> !trait.proj<@Outer[tuple<i32>], "Assoc">
  return %res : !trait.proj<@Outer[tuple<i32>], "Assoc">
}
