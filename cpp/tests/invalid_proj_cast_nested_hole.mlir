// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The unproven-cast hole check rejects a claim-application projection standing
// for a fresh variable that resolves to a spelling CONTAINING a distinct hole,
// not only one that IS a distinct hole. Crossing @C[i64]::A against a wrapper
// carrying @C[i64]::B binds hole_A to tuple<hole_B> (and, one level deeper, to
// tuple<tuple<hole_B>>); both equate two associated types the claim never
// justifies, and both are rejected.

trait.trait @C[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.trait @D[!trait.poly<1>] {}

func.func @wrapped(
  %d: !trait.claim<@D[!trait.proj<@C[i64], "A">]>,
  %c: !trait.claim<@C[i64]>
) {
  // expected-error @below {{equate distinct projections under claim}}
  %cast = trait.proj.cast %d, %c
    : !trait.claim<@D[!trait.proj<@C[i64], "A">]>
    to !trait.claim<@D[tuple<!trait.proj<@C[i64], "B">>]>
    by !trait.claim<@C[i64]>
  return
}

// -----

trait.trait @C[!trait.poly<0>] {
  trait.assoc_type @A
  trait.assoc_type @B
}

trait.trait @D[!trait.poly<1>] {}

func.func @deeper(
  %d: !trait.claim<@D[!trait.proj<@C[i64], "A">]>,
  %c: !trait.claim<@C[i64]>
) {
  // expected-error @below {{equate distinct projections under claim}}
  %cast = trait.proj.cast %d, %c
    : !trait.claim<@D[!trait.proj<@C[i64], "A">]>
    to !trait.claim<@D[tuple<tuple<!trait.proj<@C[i64], "B">>>]>
    by !trait.claim<@C[i64]>
  return
}
