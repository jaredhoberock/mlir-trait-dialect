// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Resolving an associated-type projection must substitute through the claim
// that impl selection matched against. Here the outer projection selects the
// blanket @Has impl after normalizing @Id[i32]::Out to i64. Substituting against
// the original source spelling would leave the inner projection behind.

!T = !trait.poly<0>
trait.trait @Id[!T] {
  trait.assoc_type @Out
}

trait.impl @Id_i32 for @Id[i32] {
  trait.assoc_type @Out = i64
}

trait.trait @Has[!T] {
  trait.assoc_type @Shape
}

trait.impl @Has_blanket for @Has[!T] {
  trait.assoc_type @Shape = !T
}

// CHECK-LABEL: func.func @nested_projection
// CHECK-SAME: (%{{.*}}: i64) -> i64
// CHECK-NOT: !trait.proj
func.func @nested_projection(
  %x: !trait.proj<@Has[!trait.proj<@Id[i32], "Out">], "Shape">
) -> !trait.proj<@Has[!trait.proj<@Id[i32], "Out">], "Shape"> {
  return %x : !trait.proj<@Has[!trait.proj<@Id[i32], "Out">], "Shape">
}

!U = !trait.poly<1>
trait.trait @Wrap[!T] {
  trait.assoc_type @Out<[!U]>
}

trait.impl @Wrap_i1 for @Wrap[i1] {
  trait.assoc_type @Out<[!U]> = !U
}

// CHECK-LABEL: func.func @gat_projection_arg
// CHECK-SAME: (%{{.*}}: i64) -> i64
// CHECK-NOT: !trait.proj
func.func @gat_projection_arg(
  %x: !trait.proj<@Wrap[i1], "Out", [!trait.proj<@Id[i32], "Out">]>
) -> !trait.proj<@Wrap[i1], "Out", [!trait.proj<@Id[i32], "Out">]> {
  return %x : !trait.proj<@Wrap[i1], "Out", [!trait.proj<@Id[i32], "Out">]>
}
