// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// The equality arm of trait.witness has two leaves -- proj-resolve and refl --
// and its custom assembly dispatches on which is present. This pins both
// surviving a print/parse round-trip.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A(!trait.claim<@A[!S]>) assoc {Item = 0}
trait.trait private @B(!trait.claim<@B[!S]>) assoc {Item = 0}

trait.impl private @A_impl(%self: !trait.claim<@A[!trait.poly<0>]>) bindings {Item = !trait.poly<0>} {
}
trait.impl private @B_impl(%self: !trait.claim<@B[!trait.poly<0>]>) bindings {Item = !trait.poly<0>} {
}

// CHECK-LABEL: func.func @arms
func.func @arms() {
  // CHECK: trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64] : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  %w1 = trait.witness proj_resolve !trait.proj<@A[i64], "Item"> resolves i64 by @A_impl[i64]
    : !trait.claim<!trait.proj<@A[i64], "Item"> = i64>
  // CHECK: trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl[i64] : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  %w2 = trait.witness proj_resolve !trait.proj<@B[i64], "Item"> resolves i64 by @B_impl[i64]
    : !trait.claim<!trait.proj<@B[i64], "Item"> = i64>
  // CHECK: trait.witness refl : !trait.claim<i64 = i64>
  %r = trait.witness refl : !trait.claim<i64 = i64>
  return
}
