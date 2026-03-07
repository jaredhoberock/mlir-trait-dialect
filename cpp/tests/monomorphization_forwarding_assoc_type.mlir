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
// The forwarding impl's method signature, after substituting Self=tuple<U>,
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

trait.trait @Inner[!S] {
  trait.assoc_type @Assoc
  func.func private @method(!S) -> !trait.proj<@Inner[!S], "Assoc">
}

trait.trait @Outer[!S] {
  trait.assoc_type @Assoc
  func.func private @method(!S) -> !trait.proj<@Outer[!S], "Assoc">
}

// Concrete impl: Inner for i32, Assoc = f32
trait.impl @Inner_i32 for @Inner[i32] {
  trait.assoc_type @Assoc = f32
  func.func @method(%self: i32) -> f32 {
    %c = arith.sitofp %self : i32 to f32
    return %c : f32
  }
}

// Forwarding impl: Outer for tuple<U> where Inner[U], Assoc = Inner[U]::Assoc
trait.impl @Outer_tuple for @Outer[tuple<!U>] where [@Inner[!U]] {
  trait.assoc_type @Assoc = !trait.proj<@Inner[!U], "Assoc">
  func.func @method(%self: tuple<!U>) -> !trait.proj<@Inner[!U], "Assoc"> {
    %a = trait.assume @Inner[!U]
    %elem = "test.extract"(%self) : (tuple<!U>) -> !U
    %res = trait.method.call %a @Inner[!U]::@method(%elem)
      : (!U) -> !trait.proj<@Inner[!U], "Assoc">
    return %res : !trait.proj<@Inner[!U], "Assoc">
  }
}

// Concrete call site: call Outer::method on tuple<i32>
// Outer[tuple<i32>]::Assoc -> Inner[i32]::Assoc -> f32
// CHECK-LABEL: func.func @test_forwarding
// CHECK: call @{{[^(]*}}({{.*}}) : (tuple<i32>) -> f32
// CHECK: return %{{.*}} : f32
func.func @test_forwarding(%arg: tuple<i32>) -> !trait.proj<@Outer[tuple<i32>], "Assoc"> {
  %a = trait.allege @Outer[tuple<i32>]
  %res = trait.method.call %a @Outer[tuple<i32>]::@method(%arg)
    : (tuple<i32>) -> !trait.proj<@Outer[tuple<i32>], "Assoc">
  return %res : !trait.proj<@Outer[tuple<i32>], "Assoc">
}
