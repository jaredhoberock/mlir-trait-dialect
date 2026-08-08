// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// The equality arm of trait.witness has three leaves -- proj-resolve, refl, and
// compose -- and its custom assembly dispatches on which is present. This pins
// all three surviving a print/parse round-trip byte-identically, so the compose
// leaf's addition to the dispatch leaves the proj-resolve and refl spellings
// undisturbed.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @A[!S] { trait.assoc_type @Item }
trait.trait @B[!S] { trait.assoc_type @Item }

trait.impl @A_impl for @A[!U] { trait.assoc_type @Item = !U }
trait.impl @B_impl for @B[!U] { trait.assoc_type @Item = !U }

// CHECK-LABEL: func.func @arms
func.func @arms() {
  // CHECK: trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %w1 = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl
    : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  // CHECK: trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  %w2 = trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl
    : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  // CHECK: trait.witness refl : !trait.claim<i64 = i64>
  %r = trait.witness refl : !trait.claim<i64 = i64>
  // CHECK: trait.witness compose(%{{[0-9]+}}, %{{[0-9]+}}) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>) : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>
  %c = trait.witness compose(%w1, %w2)
    : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>)
    : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>
  return
}
