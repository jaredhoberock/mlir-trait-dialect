// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The result endpoints must be a single-substitution instance of the frozen
// certificate. Here the certificate freezes `... = i64`, but the result claim's
// contractum is i32 -- a drift no substitution explains -- so verify() refuses
// it. This is the corruption today's proven branch catches by module search,
// restored as a local check.

!S = !trait.poly<0>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Output = i64
}

func.func @drift() -> !trait.claim<!trait.proj<@Trait[i64], "Output"> = i32> {
  // expected-error @below {{are not an instance of the certificate}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i32>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i32>
}
