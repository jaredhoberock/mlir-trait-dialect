// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A trait.func.call binds two type parameters independently through sibling
// projections of the same associated type: @f expects
// @D[@A[poly<4>]::Out, @A[poly<5>]::Out], and the argument is
// @D[@A[i32]::Out, @A[f32]::Out], so the call verifier unifies poly<4>=i32 and
// poly<5>=f32 by recursing through each projection.
//
// The projected argument is built one coerce per application: the equality
// @A[i32]::Out = i64 reconciles the first element, @A[f32]::Out = i64 the
// second. A coerce that respells both elements at once while citing only one of
// the two sibling equalities is refused; that negative lives in
// invalid_coerce_missing_sibling_equality.mlir.

module {
  trait.trait @D[!trait.poly<0>, !trait.poly<1>] {}
  trait.trait @A[!trait.poly<2>] { trait.assoc_type @Out }
  trait.impl @D_impl for @D[!trait.poly<3>, !trait.poly<3>] {}
  trait.impl @A_i32 for @A[i32] { trait.assoc_type @Out = i64 }
  trait.impl @A_f32 for @A[f32] { trait.assoc_type @Out = i64 }
  trait.proof @D_p proves @D_impl for @D[i64, i64] given []

  func.func nested @f(%x: !trait.poly<4>, %y: !trait.poly<5>,
    %d: !trait.claim<@D[!trait.proj<@A[!trait.poly<4>], "Out">, !trait.proj<@A[!trait.poly<5>], "Out">]>
  ) -> i32 { %0 = arith.constant 0 : i32 return %0 : i32 }

  // CHECK-LABEL: func.func @main
  // CHECK: trait.func.call @f
  func.func @main() -> i32 {
    %x = arith.constant 0 : i32
    %y = arith.constant 0.0 : f32
    %d = trait.witness @D_p for @D[i64, i64]
    // First application: coerce the first i64 to @A[i32]::Out citing @A[i32]::Out = i64.
    %eq_i32 = trait.witness proj_resolve !trait.proj<@A[i32], "Out"> resolves i64 by @A_i32
      : !trait.claim<!trait.proj<@A[i32], "Out"> = i64>
    %d1 = trait.coerce %d
      : !trait.claim<@D[i64, i64] by @D_p>
      to !trait.claim<@D[!trait.proj<@A[i32], "Out">, i64] by @D_p>
      via (%eq_i32) : (!trait.claim<!trait.proj<@A[i32], "Out"> = i64>)
    // Second application: coerce the second i64 to @A[f32]::Out citing @A[f32]::Out = i64.
    %eq_f32 = trait.witness proj_resolve !trait.proj<@A[f32], "Out"> resolves i64 by @A_f32
      : !trait.claim<!trait.proj<@A[f32], "Out"> = i64>
    %d2 = trait.coerce %d1
      : !trait.claim<@D[!trait.proj<@A[i32], "Out">, i64] by @D_p>
      to !trait.claim<@D[!trait.proj<@A[i32], "Out">, !trait.proj<@A[f32], "Out">] by @D_p>
      via (%eq_f32) : (!trait.claim<!trait.proj<@A[f32], "Out"> = i64>)
    %r = trait.func.call @f(%x, %y, %d2)
      : (i32, f32, !trait.claim<@D[!trait.proj<@A[i32], "Out">, !trait.proj<@A[f32], "Out">] by @D_p>)
      -> i32
    return %r : i32
  }
}
