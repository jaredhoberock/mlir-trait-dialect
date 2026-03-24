// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(resolve-impls-trait)' %s | FileCheck %s

// Test that resolve-impls skips polymorphic alleges (they are deferred
// to post-monomorphization) while resolving monomorphic ones.

!T = !trait.poly<0>

// CHECK: trait.trait @Safe
trait.trait @Safe [!T] {
  func.func private @check(!T) -> i32
}

trait.impl for @Safe[i32] {
  func.func @check(%x: i32) -> i32 {
    return %x : i32
  }
}

// Monomorphic allege is resolved to a witness
// CHECK: func.func @test_mono
func.func @test_mono(%x: i32) -> i32 {
  // CHECK: trait.witness
  // CHECK-NOT: trait.allege
  %c = trait.allege @Safe[i32]
  %r = trait.method.call %c @Safe[i32]::@check(%x) : (i32) -> i32
  return %r : i32
}

// Polymorphic allege is left intact (deferred).
// The allege is used by a method call to prevent DCE.
// CHECK: func.func @test_poly
!T2 = !trait.poly<2>
func.func @test_poly(%x: !T2) -> i32 {
  // CHECK: trait.allege @Safe[!trait.poly<2>] unsafe
  %c = trait.allege @Safe[!T2] unsafe
  %r = trait.method.call %c @Safe[!T2]::@check(%x) : (!T2) -> i32
  return %r : i32
}
