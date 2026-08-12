// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The witness instance check requires the current endpoints to be a single
// substitution instance of the certificate. Resolving a projection
// inside an endpoint -- collapsing @Trait[i64]::Output to i64 -- is not a
// substitution instance of the projection spelling, so the resolved endpoint
// form is refused. This is why the specializing clone rule leaves an equality
// claim's endpoints pure substitution and resolves nothing inside them.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

func.func @resolved_endpoints() -> !trait.claim<i64 = i64> {
  // expected-error @below {{result endpoints 'i64' = 'i64' are not an instance of the certificate}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[!S], "Output"> resolves !S by @Trait_impl
    : !trait.claim<i64 = i64>
  return %e : !trait.claim<i64 = i64>
}
