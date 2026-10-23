// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An equality that two proj-resolve leaves entail together, written as a coerce
// of the refl witness of @A[i64]::Item citing both, is passed to a function that
// is no template. The call lowers to a plain call whose claim still reads
// @A[i64]::Item = @B[i64]::Item, since nothing rewrites an equality's
// endpoints, and erasure's entry judgment settles it: both endpoints resolve to
// i64. Erasure removes the coerce on that judgment rather than comparing its
// written endpoints, which still differ, and the claim leaves the call and the
// callee's signature with it.

!S = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!S]>) { trait.assoc_type @Item }
trait.trait private @B(%self: !trait.claim<@B[!S]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }
trait.impl private @B_impl(%self: !trait.claim<@B[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

// CHECK-LABEL: func.func private @g() -> i64
// CHECK-NEXT: %[[C:.*]] = arith.constant 7 : i64
// CHECK-NEXT: return %[[C]] : i64
func.func private @g(%e: !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// CHECK-LABEL: func.func @use() -> i64
// CHECK-NEXT: %[[R:.*]] = call @g() : () -> i64
// CHECK-NEXT: return %[[R]] : i64
// CHECK-NOT: trait.
func.func @use() -> i64 {
  %w1 = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64] : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %w2 = trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl[i64] : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  %r0 = trait.witness refl : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
  %c = trait.coerce %r0 : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">> to !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">> via (%w1, %w2) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>)
  %r = trait.func.call @g(%c) : (!trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>) -> i64
  return %r : i64
}
