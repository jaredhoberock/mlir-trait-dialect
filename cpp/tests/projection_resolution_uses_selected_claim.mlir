// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// Resolving an associated-type projection must substitute through the claim
// that impl selection matched against. Here the outer projection selects the
// blanket @Has impl after normalizing @Id[i32]::Out to i64. Substituting against
// the original source spelling would leave the inner projection behind.

!T = !trait.poly<0>
trait.trait private @Id(!trait.claim<@Id[!T]>) assoc {Out = 0}

trait.impl private @Id_i32(%self: !trait.claim<@Id[i32]>) bindings {Out = i64} {
}

trait.trait private @Has(!trait.claim<@Has[!T]>) assoc {Shape = 0}

trait.impl private @Has_blanket(%self: !trait.claim<@Has[!T]>) bindings {Shape = !T} {
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
trait.trait private @Wrap(!trait.claim<@Wrap[!T]>) assoc {Out = 1}

trait.impl private @Wrap_i1(%self: !trait.claim<@Wrap[i1]>) bindings {Out = !trait.poly<0>} {
}

// CHECK-LABEL: func.func @gat_projection_arg
// CHECK-SAME: (%{{.*}}: i64) -> i64
// CHECK-NOT: !trait.proj
func.func @gat_projection_arg(
  %x: !trait.proj<@Wrap[i1], "Out", [!trait.proj<@Id[i32], "Out">]>
) -> !trait.proj<@Wrap[i1], "Out", [!trait.proj<@Id[i32], "Out">]> {
  return %x : !trait.proj<@Wrap[i1], "Out", [!trait.proj<@Id[i32], "Out">]>
}
