// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// trait.coerce changes a value's written type by citing type equalities. Its
// verifier runs ground congruence closure over the endpoints and the cited
// equalities. These shapes are harvested from real frontend derivations: a
// concrete/projection reconciliation, the deep congruence that lifts a leaf
// equality through a claim's trait argument, and a transitive chain.

!S = !trait.poly<0>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.trait @Bound[!S] {}

trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Output = i64
}

// Concrete <-> projection: the equality @Trait[i64]::Output = i64 reconciles the
// two spellings directly.
// CHECK-LABEL: func.func @concrete_to_projection
// CHECK: trait.coerce %arg0 : i64 to !trait.proj<@Trait[i64], "Output"> via (%0) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
func.func @concrete_to_projection(%v: i64) -> !trait.proj<@Trait[i64], "Output"> {
  %eq = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  %c = trait.coerce %v : i64 to !trait.proj<@Trait[i64], "Output">
    via (%eq) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
  return %c : !trait.proj<@Trait[i64], "Output">
}

// Deep congruence: the leaf equality lifts through @Bound's trait argument, so
// @Bound[i64] and @Bound[@Trait[i64]::Output] denote one claim.
// CHECK-LABEL: func.func @deep_congruence
// CHECK: trait.coerce %{{.*}} : !trait.claim<@Bound[i64]> to !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]>
func.func @deep_congruence() -> !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]> {
  %b = trait.allege @Bound[i64]
  %eq = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[i64]> to !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]>
    via (%eq) : (!trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>)
  return %c : !trait.claim<@Bound[!trait.proj<@Trait[i64], "Output">]>
}

// Transitive chain: two cited equalities chain through the union-find.
// CHECK-LABEL: func.func @chain
// CHECK: trait.coerce %arg0 : i64 to i16 via (%arg1, %arg2)
func.func @chain(%v: i64,
                 %e0: !trait.claim<i64 = i32>,
                 %e1: !trait.claim<i32 = i16>) -> i16 {
  %c = trait.coerce %v : i64 to i16
    via (%e0, %e1) : (!trait.claim<i64 = i32>, !trait.claim<i32 = i16>)
  return %c : i16
}
