// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Test that `trait.allege ... unsafe` round-trips through parse/print
// and passes verification for polymorphic claims.

!T = !trait.poly<0>

trait.trait @Safe [!T] {}

trait.impl for @Safe[i32] {}

// Polymorphic allege with unsafe passes verification
// CHECK: func.func @test_unsafe_allege
func.func @test_unsafe_allege(%x: i32) -> i32 {
  // CHECK: trait.allege @Safe[!trait.poly<0>] unsafe
  %c = trait.allege @Safe[!T] unsafe
  return %x : i32
}

// Monomorphic allege without unsafe still works
// CHECK: func.func @test_normal_allege
func.func @test_normal_allege(%x: i32) -> i32 {
  // CHECK: trait.allege @Safe[i32]
  // CHECK-NOT: unsafe
  %c = trait.allege @Safe[i32]
  return %x : i32
}
