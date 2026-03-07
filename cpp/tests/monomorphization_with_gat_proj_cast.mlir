// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Test that GATs work when the trait and impl use different poly IDs
// for the GAT type parameter. This matches what todolang's codegen produces.
//
// trait Trait {
//   type Assoc<U>;
// }
// impl Trait for Number {
//   type Assoc<U> = U;
// }
// fn foo<T: Trait, V>(arg: T, value: T::Assoc<V>) -> T::Assoc<V> { value }

// Trait uses poly<0> for Self, poly<1> for GAT param
trait.trait @Trait[!trait.poly<0>] {
  trait.assoc_type @Assoc<[!trait.poly<1>]>
  func.func private @method(!trait.poly<0>) -> !trait.proj<@Trait[!trait.poly<0>], "Assoc", [!trait.poly<1>]>
}

// Impl uses poly<99> for GAT param (different from trait's poly<1>)
trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Assoc<[!trait.poly<99>]> = !trait.poly<99>
  func.func @method(%self: i64) -> i64 {
    return %self : i64
  }
}

// A generic function that takes a projection-typed arg
func.func @foo(%arg: !trait.poly<0>, %value: !trait.proj<@Trait[!trait.poly<0>], "Assoc", [!trait.poly<2>]>,
               %claim: !trait.claim<@Trait[!trait.poly<0>]>)
    -> !trait.proj<@Trait[!trait.poly<0>], "Assoc", [!trait.poly<2>]> {
  return %value : !trait.proj<@Trait[!trait.poly<0>], "Assoc", [!trait.poly<2>]>
}

// CHECK-LABEL: func.func @caller
// CHECK-NOT: !trait.proj
// CHECK: return %{{.*}} : i1
func.func @caller() -> i1 {
  %x = arith.constant 7 : i64
  %v = arith.constant true
  %w = trait.witness @Trait_impl for @Trait[i64]
  // Cast i1 to Trait[i64]::Assoc<i1> via proj.cast
  %pw = trait.proj.cast %v, %w : i1 to !trait.proj<@Trait[i64], "Assoc" by @Trait_impl, [i1]>
  %r = trait.func.call @foo(%x, %pw, %w) : (i64, !trait.proj<@Trait[i64], "Assoc" by @Trait_impl, [i1]>, !trait.claim<@Trait[i64] by @Trait_impl>) -> i1
  return %r : i1
}
