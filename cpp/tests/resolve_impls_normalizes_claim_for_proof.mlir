// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Impl selection normalizes monomorphic projections in the wanted claim before
// candidate matching. Proof construction and obligation specialization must
// use that same normalized claim: the two projected arguments below both
// resolve to i64, but their source spellings are different and would not both
// bind to the repeated !T in @Eq.

!L = !trait.poly<0>
!R = !trait.poly<1>
trait.trait @Eq[!L, !R] {
  func.func nested @use()
}

trait.trait @Ord[!trait.poly<4>] {}

trait.impl @Ord_i64 for @Ord[i64] {}

!T = !trait.poly<2>
trait.impl @Eq_same for @Eq[!T, !T] where [@Ord[!T]] {
  func.func nested @use() {
    return
  }
}

!A = !trait.poly<3>
trait.trait @Has[!A] {
  trait.assoc_type @Shape
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Shape = i64
}

trait.impl @Has_f32 for @Has[f32] {
  trait.assoc_type @Shape = i64
}

func.func @repro() {
  // CHECK-NOT: trait.allege
  // CHECK: call @Eq_same_{{.*}}_use()
  %claim = trait.allege @Eq[
    !trait.proj<@Has[i32], "Shape">,
    !trait.proj<@Has[f32], "Shape">
  ]
  trait.method.call %claim @Eq[
    !trait.proj<@Has[i32], "Shape">,
    !trait.proj<@Has[f32], "Shape">
  ]::@use() : () -> ()
  return
}
