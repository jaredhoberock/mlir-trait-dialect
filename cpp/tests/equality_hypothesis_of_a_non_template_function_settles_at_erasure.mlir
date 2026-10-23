// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A function that is no template assumes @A[i64]::Item = i64 and returns it
// beside a value. The hypothesis holds: the one impl of @A binds Item to its
// argument. Erasure's entry resolves the projection, finds both sides i64, and
// lets the hypothesis leave with the parameter, whether instantiation ran first
// or not.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

// CHECK-LABEL: func.func @use() -> i64
// CHECK-NEXT: %[[C:.*]] = arith.constant 7 : i64
// CHECK-NEXT: return %[[C]] : i64
// CHECK-NOT: trait.
func.func @use(%h: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> (i64, !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) {
  %v = arith.constant 7 : i64
  return %v, %h : i64, !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
}
