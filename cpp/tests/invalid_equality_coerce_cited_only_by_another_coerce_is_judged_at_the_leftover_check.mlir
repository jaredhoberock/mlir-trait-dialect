// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The leftover check judges every monomorphic equality a coerce states, whatever
// reads it. %c derives i64 = i32 from the hypothesis i32 = i64, which no caller
// supplies, and only the coerce %d cites it; %d's own equality holds through %w
// alone and settles. Erasure removes a coerce to an equality without comparing
// its endpoints, so the leftover check is where %c's equality is refused: its
// endpoints are two ground types.

!S = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!S]>) { trait.assoc_type @Item }
trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

func.func private @g(%e: !trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

func.func @use(%h: !trait.claim<i32 = i64>) -> i64 {
  %w = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64] : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %r0 = trait.witness refl : !trait.claim<i64 = i64>
  // CHECK: error: unproven monomorphic claim '!trait.claim<i64 = i32>' after instantiate-monomorphs
  %c = trait.coerce %r0 : !trait.claim<i64 = i64> to !trait.claim<i64 = i32> via (%h) : (!trait.claim<i32 = i64>)
  %r1 = trait.witness refl : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
  %d = trait.coerce %r1 : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">> to !trait.claim<!trait.proj<@A[i64], "Item"> = i64> via (%w, %c) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<i64 = i32>)
  %r = trait.func.call @g(%d) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>) -> i64
  return %r : i64
}
