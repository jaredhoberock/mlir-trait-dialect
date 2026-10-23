// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The true twin of invalid_projection_in_a_proven_claims_argument_is_refused_at_erasure.mlir:
// @A_i64 resolves @A[i64]::Item to i64, so the projection @B_proj's claim
// states is a type, erasure's entry settles it, and the evidence leaves with
// erasure on either route, the call keeping the value its callee computes.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) { trait.assoc_type @Item = i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @B_proj(%self: !trait.claim<@B[!trait.proj<@A[i64], "Item">]>) {}

// CHECK-LABEL: func.func @use() -> i64
// CHECK: arith.constant 7 : i64
func.func @use(%w: !trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// CHECK-LABEL: func.func @run() -> i64
// CHECK-NEXT: %[[V:.*]] = call @use() : () -> i64
// CHECK-NEXT: return %[[V]] : i64
func.func @run() -> i64 {
  %w = trait.witness @B_proj for @B[!trait.proj<@A[i64], "Item">]
  %v = func.call @use(%w) : (!trait.claim<@B[!trait.proj<@A[i64], "Item">] by @B_proj>) -> i64
  return %v : i64
}
