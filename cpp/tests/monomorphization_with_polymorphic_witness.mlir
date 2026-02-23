// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Verifies that monomorphization specializes through polymorphic witnesses.

!T0 = !trait.poly<0>
!T1 = !trait.poly<1>

trait.trait @Trait [!T0, !T1] {
  func.func private @method(!T0, !T1) -> i64
}

!T2 = !trait.poly<2>
trait.impl @Trait_impl for @Trait[i64, !T2] {
  func.func @method(%self: i64, %arg: !T2) -> i64 {
    %0 = trait.assume @Trait[i64, !T2]
    return %self : i64
  }
}

!T3 = !trait.poly<3>
trait.proof @Trait_proof proves @Trait_impl for @Trait[i64, tuple<!T3>] given []

// CHECK-LABEL: func.func @test_
// CHECK: call @Trait_impl_{{.*}}_method
func.func @test(%c: !trait.claim<@Trait[i64, tuple<!T3>]>, %arg0: tuple<!T3>) -> i64 {
  %c0 = arith.constant 0 : i64
  %1 = trait.method.call %c @Trait[i64, tuple<!T3>]::@method(%c0, %arg0)
    : (i64, tuple<!T3>) -> i64
  return %1 : i64
}

// CHECK-LABEL: func.func @main
// CHECK: call @test_
func.func @main(%t: tuple<i32>) -> i64 {
  %p = trait.witness @Trait_proof for @Trait[i64, tuple<i32>]
  %res = trait.func.call @test(%p, %t)
    : (!trait.claim<@Trait[i64, tuple<i32>] by @Trait_proof>, tuple<i32>) -> i64
  return %res : i64
}
