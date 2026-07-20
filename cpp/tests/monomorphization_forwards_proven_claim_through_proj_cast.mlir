// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' %s | FileCheck %s

// A generic function forwards a proven trait claim into a projection-spelled
// bound via trait.proj.cast. The frontend emits the cast's result claim
// unproven by construction (see the ProjCastOp description in TraitOps.td); its
// third type argument is @Fold[i1]::Item rather than the concrete type the input
// already proves.
//
// When @gen_fold is specialized for F = i32, projection resolution rewrites
// @Fold[i1]::Item to i64, at which point the cast's proven input claim
// @FoldFn[i32, i64, i64] and its unproven result claim name the same trait
// application, differing only in the proof. InheritProjCastProofPattern must
// then retype the result to the input's proven type so the identity cast folds
// away and the downstream method call resolves to the concrete impl. Without
// it, the unproven monomorphic claim survives to the leftover check.

// CHECK-NOT: trait.trait
// CHECK-NOT: trait.impl
// CHECK-NOT: trait.proof
// CHECK-NOT: trait.proj.cast
// CHECK-NOT: !trait.proj
// CHECK-NOT: !trait.claim

!F = !trait.poly<0>

// The trait forwarded and proven; specialization discharges its i32 impl.
trait.trait @FoldFn[!trait.poly<0>, !trait.poly<1>, !trait.poly<2>] {
  func.func private @apply(!trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>
}

// The trait whose associated type supplies the projection in the bound.
trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

trait.impl @FoldFn_i32 for @FoldFn[i32, i64, i64] {
  func.func @apply(%f: i32, %a: i64, %b: i64) -> i64 {
    %r = arith.addi %a, %b : i64
    return %r : i64
  }
}

trait.impl @Fold_i1 for @Fold[i1] {
  trait.assoc_type @Item = i64
}

// Bound projection-spelled: forward @FoldFn[F, i64, i64] into
// @FoldFn[F, i64, @Fold[i1]::Item], then invoke the callable through the cast.
func.func @gen_fold(%f: !F, %claim: !trait.claim<@FoldFn[!F, i64, i64]>) -> i64 {
  %a = arith.constant 10 : i64
  %b = arith.constant 5 : i64
  %fold_c = trait.allege @Fold[i1]
  %cast = trait.proj.cast %claim, %fold_c
    : !trait.claim<@FoldFn[!F, i64, i64]>
    to !trait.claim<@FoldFn[!F, i64, !trait.proj<@Fold[i1], "Item">]>
    by !trait.claim<@Fold[i1]>
  %r = trait.method.call %cast @FoldFn[!F, i64, !trait.proj<@Fold[i1], "Item">]::@apply(%f, %a, %b)
    : (!F, i64, i64) -> i64
  return %r : i64
}

// CHECK-LABEL: func.func @caller
// CHECK: call @gen_fold
// CHECK: return %{{.*}} : i64
func.func @caller() -> i64 {
  %f = arith.constant 7 : i32
  %w = trait.witness @FoldFn_i32 for @FoldFn[i32, i64, i64]
  %r = trait.func.call @gen_fold(%f, %w) : (i32, !trait.claim<@FoldFn[i32, i64, i64] by @FoldFn_i32>) -> i64
  return %r : i64
}
