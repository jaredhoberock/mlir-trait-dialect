// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// Verifies that a polymorphic claim can carry a proof when the impl
// covers all types (e.g. impl<T> Trait<T> for Number).

!T0 = !trait.poly<0>
!T1 = !trait.poly<1>

// CHECK: trait.trait @Trait
trait.trait @Trait [!T0, !T1] {
  func.func private @method(!T0, !T1) -> i64
}

!T2 = !trait.poly<2>
// CHECK: trait.impl @Trait_impl for @Trait[i64, !trait.poly<2>]
trait.impl @Trait_impl for @Trait[i64, !T2] {
  func.func @method(%self: i64, %arg: !T2) -> i64 {
    %0 = trait.assume @Trait[i64, !T2]
    return %self : i64
  }
}

!T3 = !trait.poly<3>

// CHECK: trait.proof @Trait_proof proves @Trait_impl for @Trait[i64, tuple<!trait.poly<3>>] given []
trait.proof @Trait_proof proves @Trait_impl for @Trait[i64, tuple<!T3>] given []

// CHECK: func.func @test
func.func @test(%arg0: tuple<!T3>) -> i64 {
  // CHECK: trait.witness @Trait_proof for @Trait[i64, tuple<!trait.poly<3>>]
  %0 = trait.witness @Trait_proof for @Trait[i64, tuple<!T3>]
  %c0 = arith.constant 0 : i64
  %1 = trait.method.call %0 @Trait[i64, tuple<!T3>]::@method(%c0, %arg0)
    : (i64, tuple<!T3>) -> i64
    by @Trait_proof
  return %1 : i64
}
