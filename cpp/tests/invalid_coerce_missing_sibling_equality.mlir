// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A coerce that respells both parameters of @D at once must cite an equality for
// each: the two sibling projections @A[i32]::Out and @A[f32]::Out are distinct
// applications, and evidence for one says nothing about the other. Citing only
// @A[i32]::Out = i64 leaves the second element @A[f32]::Out uncongruent with the
// i64 the input carries there, so the endpoints are not equal under the cited
// equalities and the coerce is refused. This is the shape a frontend produces
// when it drops one of two required sibling equalities; the accept path that
// cites both, one coerce per application, lives in
// proven_claim_sibling_projection_unification.mlir.

module {
  trait.trait @D[!trait.poly<0>, !trait.poly<1>] {}
  trait.trait @A[!trait.poly<2>] { trait.assoc_type @Out }
  trait.impl @D_impl for @D[!trait.poly<3>, !trait.poly<3>] {}
  trait.impl @A_i32 for @A[i32] { trait.assoc_type @Out = i64 }
  trait.impl @A_f32 for @A[f32] { trait.assoc_type @Out = i64 }
  trait.proof @D_p proves @D_impl for @D[i64, i64] given []

  func.func @main() -> i32 {
    %d = trait.witness @D_p for @D[i64, i64]
    %eq_i32 = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i64 by @A_i32
      : !trait.claim<!trait.proj<@A[i32], "Out"> = i64>
    // expected-error @below {{are not equal under the cited equalities}}
    %d1 = trait.coerce %d
      : !trait.claim<@D[i64, i64] by @D_p>
      to !trait.claim<@D[!trait.proj<@A[i32], "Out">, !trait.proj<@A[f32], "Out">] by @D_p>
      via (%eq_i32) : (!trait.claim<!trait.proj<@A[i32], "Out"> = i64>)
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
