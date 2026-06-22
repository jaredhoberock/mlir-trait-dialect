// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s 2>&1 | FileCheck %s

// Normalization must not make a repeated type parameter match projections that
// resolve to different ground types.

!L = !trait.poly<0>
!R = !trait.poly<1>
trait.trait @Eq[!L, !R] {
  func.func nested @use()
}

!T = !trait.poly<2>
trait.impl @Eq_same for @Eq[!T, !T] {
  func.func nested @use() {
    return
  }
}

!A = !trait.poly<3>
trait.trait @Has[!A] {
  trait.assoc_type @Shape
}

trait.impl @Has_i32 for @Has[i32] {
  trait.assoc_type @Shape = i32
}

trait.impl @Has_f32 for @Has[f32] {
  trait.assoc_type @Shape = i64
}

func.func @repro() {
  // CHECK: no impl with satisfiable assumptions for '!trait.claim<@Eq
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
