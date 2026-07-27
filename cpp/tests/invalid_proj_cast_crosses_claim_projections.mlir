// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A trait.proj.cast justified by an unproven claim may resolve only the
// projections over the claim's own trait application, each standing for a fresh
// variable. Crossing two distinct such projections -- here @C[i64]::A on one
// side against @C[i64]::B on the other -- binds one hole to the other, equating
// two associated types the claim never justifies. The cast is rejected.

module {
  trait.trait @C[!trait.poly<0>] {
    trait.assoc_type @A
    trait.assoc_type @B
  }

  trait.trait @D[!trait.poly<1>, !trait.poly<2>] {}

  func.func @f(
    %d: !trait.claim<@D[!trait.proj<@C[i64], "A">, !trait.proj<@C[i64], "B">]>,
    %c: !trait.claim<@C[i64]>
  ) {
    // expected-error @below {{equate distinct projections under claim}}
    %cast = trait.proj.cast %d, %c
      : !trait.claim<@D[!trait.proj<@C[i64], "A">, !trait.proj<@C[i64], "B">]>
      to !trait.claim<@D[!trait.proj<@C[i64], "B">, !trait.proj<@C[i64], "A">]>
      by !trait.claim<@C[i64]>
    return
  }
}
