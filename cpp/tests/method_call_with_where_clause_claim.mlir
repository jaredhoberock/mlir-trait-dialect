// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Trait @SameAs with a convert method
trait.trait @SameAs[!trait.poly<0>, !trait.poly<1>] {
  func.func nested @convert(!trait.poly<1>) -> !trait.poly<0>
}
trait.impl @SameAs_i32_i32 for @SameAs[i32, i32] {
  func.func nested @convert(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}

// Trait @Chooser whose method has a where-clause requiring @SameAs
!T = !trait.poly<2>
!U = !trait.poly<3>
trait.trait @Chooser[!T] {
  func.func nested @choose(!T, !U, !trait.claim<@SameAs[!T, !U]>) -> !T
}
trait.impl @Chooser_i32 for @Chooser[i32] {
  func.func nested @choose(%a: i32, %b: i32, %same: !trait.claim<@SameAs[i32, i32]>) -> i32 {
    // inner method call uses the where-clause claim
    %converted = trait.method.call %same @SameAs[i32, i32]::@convert(%b)
      : (i32) -> i32
    return %converted : i32
  }
}

func.func @test(%x: i32, %y: i32) -> i32 {
  %chooser = trait.allege @Chooser[i32]
  %same = trait.allege @SameAs[i32, i32]
  %res = trait.method.call %chooser @Chooser[i32]::@choose(%x, %y, %same)
    : (i32, i32, !trait.claim<@SameAs[i32, i32]>) -> i32
  return %res : i32
}

// Instance of SameAs::convert
// CHECK: func.func private @SameAs_i32_i32_convert(

// The instantiated choose should call SameAs_i32_i32_convert
// CHECK: func.func private @Chooser_i32_choose(
// CHECK: call @SameAs_i32_i32_convert

// Top-level test calls Chooser_i32_choose
// CHECK: func.func @test(
// CHECK: call @Chooser_i32_choose

// No trait ops should remain
// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.allege
