// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A premise may cite a conditional impl only when the citing impl's own
// where-clause covers that impl's assumptions -- hypothetical discharge, not
// module scavenging. @Sib_i64_cond binds Sib[i64]::Elem = i32 but assumes
// @X[i64]; the citing impl @Host_i64 declares no assumptions, so the audit's
// obligation check finds @X[i64] undischarged and refuses the impl at birth.

!S = !trait.poly<0>

trait.trait @X[!S] {}

trait.trait @Sib[!S] {
  trait.assoc_type @Elem
}

trait.impl @Sib_i64_cond for @Sib[i64] where [@X[i64]] {
  trait.assoc_type @Elem = i32
}

trait.trait @Host[!S] {
  func.func private @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// expected-error @below {{cited impl '@Sib_i64_cond' has an undischarged assumption '!trait.claim<@X[i64]>'; the witness premises do not supply it}}
trait.impl @Host_i64 for @Host[i64]
    premises [#trait<certificate !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond>] {
  func.func @make(%x: i64) -> i32 {
    %r = ub.poison : i32
    return %r : i32
  }
}
