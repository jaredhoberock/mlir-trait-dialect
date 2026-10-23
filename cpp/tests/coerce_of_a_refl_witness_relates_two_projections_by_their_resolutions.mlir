// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A coerce of a refl witness builds a multi-hop equality from the equality
// claims it cites without alleging it. Two proj-resolve witnesses establish
// @A[i64]::Item = i64 and @B[i64]::Item = i64; a coerce of the refl witness of
// @A[i64]::Item citing both names the equality their ground congruence closure
// entails, @A[i64]::Item = @B[i64]::Item, because both projections unite with
// i64 and hence with each other. The coerce stores only the two leaves; the
// transitivity that carries them to the result is re-derived at verify.

// VERIFY: %[[LHS:[0-9]+]] = trait.witness refl : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
// VERIFY: trait.coerce %[[LHS]] : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">> to !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">> via (%{{[0-9]+}}, %{{[0-9]+}}) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>)

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A(%self: !trait.claim<@A[!S]>) { trait.assoc_type @Item }
trait.trait private @B(%self: !trait.claim<@B[!S]>) { trait.assoc_type @Item }

trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }
trait.impl private @B_impl(%self: !trait.claim<@B[!trait.poly<0>]>) { trait.assoc_type @Item = !trait.poly<0> }

func.func @use(%pa: !trait.proj<@A[i64], "Item">) -> !trait.proj<@B[i64], "Item"> {
  %w1 = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64]
    : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %w2 = trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl[i64]
    : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  %r = trait.witness refl
    : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
  %c = trait.coerce %r
    : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@A[i64], "Item">>
    to !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>
    via (%w1, %w2) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>)
  %v = trait.coerce %pa : !trait.proj<@A[i64], "Item"> to !trait.proj<@B[i64], "Item"> via (%c)
    : (!trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>)
  return %v : !trait.proj<@B[i64], "Item">
}

// Monomorphizing resolves both concrete projections to i64: the value coerce
// the equality justifies becomes an identity and folds, and the equality's
// coerce and the witnesses it cites die by DCE. The concrete instance is a clean
// identity function carrying no trait op.

// CHECK-LABEL: func.func @use(%arg0: i64) -> i64
// CHECK-NEXT: return %arg0 : i64
// CHECK-NOT: trait.witness
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.claim
