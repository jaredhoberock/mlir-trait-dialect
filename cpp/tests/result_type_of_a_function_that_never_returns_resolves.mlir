// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The true twin of invalid_result_type_of_a_function_that_never_returns_is_judged.mlir:
// @A_i64 resolves @A[i64]::Item to i64, so the claim the declared result type
// states is about a type, erasure's entry settles it, and the result leaves
// with erasure on either route.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) { trait.assoc_type @Item = i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @I(%self: !trait.claim<@B[!trait.proj<@A[i64], "Item">]>) {}

// CHECK-LABEL: func.func @f() {
// CHECK-NEXT: cf.br ^bb1
func.func @f() -> !trait.claim<@B[!trait.proj<@A[i64], "Item">] by @I> {
  cf.br ^loop
^loop:
  cf.br ^loop
}
