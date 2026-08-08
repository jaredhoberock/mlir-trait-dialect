// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The composition leaf of trait.witness builds a multi-hop equality from its
// equality-claim premises without alleging it. Two proj-resolve certificates
// establish @A[i64]::Item = i64 and @B[i64]::Item = i64; composing them names
// the equality their ground congruence closure entails, @A[i64]::Item =
// @B[i64]::Item, because both projections unite with i64 and hence with each
// other. The witness stores only the two leaf premises; the transitivity that
// carries them to the result is re-derived at verify.

// VERIFY: trait.witness compose(%{{[0-9]+}}, %{{[0-9]+}}) : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>) : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @A[!S] { trait.assoc_type @Item }
trait.trait @B[!S] { trait.assoc_type @Item }

trait.impl @A_impl for @A[!U] { trait.assoc_type @Item = !U }
trait.impl @B_impl for @B[!U] { trait.assoc_type @Item = !U }

func.func @use(%pa: !trait.proj<@A[i64], "Item">) -> !trait.proj<@B[i64], "Item"> {
  %w1 = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl
    : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %w2 = trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl
    : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  %c = trait.witness compose(%w1, %w2)
    : (!trait.claim<!trait.proj<@A[i64], "Item"> = i64>, !trait.claim<!trait.proj<@B[i64], "Item"> = i64>)
    : !trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>
  %v = trait.coerce %pa : !trait.proj<@A[i64], "Item"> to !trait.proj<@B[i64], "Item"> via (%c)
    : (!trait.claim<!trait.proj<@A[i64], "Item"> = !trait.proj<@B[i64], "Item">>)
  return %v : !trait.proj<@B[i64], "Item">
}

// Monomorphizing resolves both concrete projections to i64: the composition
// witness's endpoints ground-resolve to one spelling and settle like any
// equality claim, the coerce they justify becomes an identity and folds, and the
// composition and proj-resolve witnesses die by DCE. The concrete instance is a
// clean identity function carrying no trait op.

// CHECK-LABEL: func.func @use(%arg0: i64) -> i64
// CHECK-NEXT: return %arg0 : i64
// CHECK-NOT: trait.witness
// CHECK-NOT: trait.coerce
// CHECK-NOT: trait.claim
