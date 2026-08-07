// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// Stripping receipts before comparison equates proofs, never types. A cast whose
// input asserts a binding the named impl contradicts -- @Safe[i32, i32] against
// a result that resolves @Conv[i1]::At to i64 -- is still a genuine resolved-type
// mismatch, and the proven judgment still rejects it. The bridge that lets a
// receipt crumb through does not let a wrong binding through with it.

!S = !trait.poly<0>

trait.trait @Conv[!S] {
  trait.assoc_type @At
}

trait.trait @Safe[!trait.poly<0>, !trait.poly<1>] {
}

trait.impl @Conv_i1 for @Conv[i1] {
  trait.assoc_type @At = i64
}

trait.impl @Safe_i32_i32 for @Safe[i32, i32] {
}

trait.proof @Conv_proof proves @Conv_i1 for @Conv[i1] given []
trait.proof @Safe_i32_i32_proof proves @Safe_i32_i32 for @Safe[i32, i32] given []

func.func @wrong_binding() -> !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]> {
  %safe = trait.witness @Safe_i32_i32_proof for @Safe[i32, i32]
  %conv = trait.witness @Conv_proof for @Conv[i1]
  // expected-error @below {{does not match resolved result type}}
  %cast = trait.proj.cast %safe, %conv
    : !trait.claim<@Safe[i32, i32] by @Safe_i32_i32_proof>
    to !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
    by !trait.claim<@Conv[i1] by @Conv_proof>
  return %cast : !trait.claim<@Safe[i32, !trait.proj<@Conv[i1], "At">]>
}
