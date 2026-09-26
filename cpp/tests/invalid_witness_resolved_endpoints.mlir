// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A projection-resolution witness's result is its own equality. A result whose
// endpoints differ from the witness's -- here an instance with the projection
// collapsed to i64 -- is refused: a clone rebuilds the witness and respells the
// claim under one substitution, and resolves nothing inside either.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl private @Trait_impl for @Trait[!U] {
  trait.assoc_type @Output = !U
}

func.func @resolved_endpoints() -> !trait.claim<i64 = i64> {
  // expected-error @below {{result endpoints 'i64' = 'i64' are not the witness's}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[!S], "Output"> resolves !S by @Trait_impl[!U = !S]
    : !trait.claim<i64 = i64>
  return %e : !trait.claim<i64 = i64>
}
