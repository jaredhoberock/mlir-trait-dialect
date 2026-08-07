// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// The clone rule specializes an equality claim's endpoints while the frozen
// certificate stays immutable. This pins the post-clone state: the certificate
// is frozen at the general spelling (over !trait.poly<0>), and the result claim
// is a specialized instance (i64). verify() accepts it because the current
// endpoints are a single-substitution instance of the frozen ones, and the seam
// audit accepts it because the cited generic impl binds the general redex to the
// general contractum. This is the invariant the bespoke clone rule maintains
// when a surrounding value specializes.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

// CHECK-LABEL: func.func @cloned
// CHECK: trait.witness proj_resolve !trait.proj<@Trait[!trait.poly<0>], "Output"> resolves !trait.poly<0> by @Trait_impl : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
func.func @cloned() -> !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64> {
  %e = trait.witness proj_resolve !trait.proj<@Trait[!trait.poly<0>], "Output"> resolves !trait.poly<0> by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
}
