// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A proj.cast between projections of two different traits is justified only for
// the projection over its claim's own application. @A[i64]::Assoc and
// @B[i64]::Assoc both resolve to i1, but a single @B[i64] claim justifies only
// the @B side; it says nothing about @A[i64]::Assoc. That input projection
// survives resolution and no longer matches the i1 the result resolves to, so
// the cast is rejected -- a coincidental cross-trait equality needs a claim for
// each application, which one proj.cast cannot supply.

!T = !trait.poly<0>

trait.trait @A[!T] {
  trait.assoc_type @Assoc
}

trait.trait @B[!T] {
  trait.assoc_type @Assoc
}

trait.impl @A_i64 for @A[i64] {
  trait.assoc_type @Assoc = i1
}

trait.impl @B_i64 for @B[i64] {
  trait.assoc_type @Assoc = i1
}

trait.proof @A_proof proves @A_i64 for @A[i64] given []
trait.proof @B_proof proves @B_i64 for @B[i64] given []

func.func @proj_to_proj(%a_proj: !trait.proj<@A[i64], "Assoc">) -> !trait.proj<@B[i64], "Assoc"> {
  %b_claim = trait.witness @B_proof for @B[i64]

  // expected-error @below {{does not match resolved result type}}
  %b_proj = trait.proj.cast %a_proj, %b_claim
    : !trait.proj<@A[i64], "Assoc"> to !trait.proj<@B[i64], "Assoc"> by !trait.claim<@B[i64] by @B_proof>

  return %b_proj : !trait.proj<@B[i64], "Assoc">
}
