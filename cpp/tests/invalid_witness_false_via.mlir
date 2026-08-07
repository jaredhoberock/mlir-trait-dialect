// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// The seam audit refuses a certificate that cites an impl which does not bind
// as stated. Here @Trait_impl binds @Trait[i64]::Output to i32, so a
// certificate claiming it resolves to i64 is a false citation.

!S = !trait.poly<0>

trait.trait @Trait[!S] {
  trait.assoc_type @Output
}

trait.impl @Trait_impl for @Trait[i64] {
  trait.assoc_type @Output = i32
}

func.func @false_via() -> !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64> {
  // expected-error @below {{binds the redex to 'i32', not the certified contractum 'i64'}}
  %e = trait.witness proj_resolve !trait.proj<@Trait[i64], "Output"> resolves i64 by @Trait_impl
    : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
  return %e : !trait.claim<!trait.proj<@Trait[i64], "Output"> = i64>
}
