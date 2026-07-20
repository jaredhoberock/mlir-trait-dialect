// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' %s | FileCheck %s

// InheritProjCastProofPattern copies a proj.cast's proven input proof onto its
// unproven result. It fires only when ALL three of its guards hold; each
// function below pins one guard by violating it and checking the cast is left
// alone:
//   * the input claim must be PROVEN -- an unproven input carries no proof to
//     inherit (@tpl_input_unproven);
//   * the result claim must be UNPROVEN -- an already-proven result must not be
//     overwritten with a different proof (@tpl_result_already_proven);
//   * the input and result must name the SAME trait application -- a proj.cast
//     never changes which impl proves a claim, so differing applications are
//     distinct logical claims (@tpl, below).
//
// Runs `instantiate-monomorphs-trait` rather than the full pipeline because the
// full pipeline prunes these never-instantiated templates, erasing the very
// proj.casts the cases inspect.

!F = !trait.poly<0>

trait.trait @FoldFn[!trait.poly<0>, !trait.poly<1>, !trait.poly<2>] {
  func.func private @apply(!trait.poly<0>, !trait.poly<1>, !trait.poly<2>) -> !trait.poly<1>
}

trait.trait @Fold[!trait.poly<0>] {
  trait.assoc_type @Item
}

trait.impl @FoldFn_i32 for @FoldFn[i32, i64, i64] {
  func.func @apply(%f: i32, %a: i64, %b: i64) -> i64 {
    %r = arith.addi %a, %b : i64
    return %r : i64
  }
}

// A second, distinct impl of the same application. It supplies a proof symbol
// different from @FoldFn_i32 so @tpl_result_already_proven can present a result
// proven by a proof the input does not name.
trait.impl @FoldFn_i32_alt for @FoldFn[i32, i64, i64] {
  func.func @apply(%f: i32, %a: i64, %b: i64) -> i64 {
    %r = arith.muli %a, %b : i64
    return %r : i64
  }
}

// Mismatched applications: proven input @FoldFn[i32, i64, i64], unproven result
// @FoldFn[i32, i64, @Fold[F]::Item]. @tpl is never specialized, so F stays
// polymorphic and the projection never resolves; the two applications remain
// distinct and the pattern must decline, leaving the projection-spelled result.

// CHECK-LABEL: func.func @tpl
// CHECK: trait.proj.cast %{{.*}}, %{{.*}} : !trait.claim<@FoldFn[i32, i64, i64] by @FoldFn_i32> to !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!trait.poly<0>], "Item">]> by
func.func @tpl(%f: !F, %fold_c: !trait.claim<@Fold[!F]>)
    -> !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]> {
  %w = trait.witness @FoldFn_i32 for @FoldFn[i32, i64, i64]
  %cast = trait.proj.cast %w, %fold_c
    : !trait.claim<@FoldFn[i32, i64, i64] by @FoldFn_i32>
    to !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]>
    by !trait.claim<@Fold[!F]>
  return %cast : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]>
}

// Unproven input: input and result name the same application
// @FoldFn[i32, i64, @Fold[F]::Item], but the input claim carries no proof. The
// pattern must decline -- there is no proof to inherit -- so the result stays
// unproven. (Input and result coincide, so the identity folder collapses the
// cast; the returned claim carries no `by`, confirming no proof was minted.)

// CHECK-LABEL: func.func @tpl_input_unproven
// CHECK: return %{{.*}} : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!trait.poly<0>], "Item">]>
func.func @tpl_input_unproven(%f: !F, %fold_c: !trait.claim<@Fold[!F]>,
    %x: !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]>)
    -> !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]> {
  %cast = trait.proj.cast %x, %fold_c
    : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]>
    to !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]>
    by !trait.claim<@Fold[!F]>
  return %cast : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">]>
}

// Result already proven: input and result name the same application
// @FoldFn[i32, i64, @Fold[F]::Item], the input is proven by @FoldFn_i32, and
// the result is already proven by the distinct @FoldFn_i32_alt. The pattern must
// decline -- overwriting the result's proof would silently swap which impl
// proves it -- so the result keeps its own @FoldFn_i32_alt.

// CHECK-LABEL: func.func @tpl_result_already_proven
// CHECK: trait.proj.cast %{{.*}}, %{{.*}} : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!trait.poly<0>], "Item">] by @FoldFn_i32> to !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!trait.poly<0>], "Item">] by @FoldFn_i32_alt> by
func.func @tpl_result_already_proven(%f: !F, %fold_c: !trait.claim<@Fold[!F]>,
    %x: !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">] by @FoldFn_i32>)
    -> !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">] by @FoldFn_i32_alt> {
  %cast = trait.proj.cast %x, %fold_c
    : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">] by @FoldFn_i32>
    to !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">] by @FoldFn_i32_alt>
    by !trait.claim<@Fold[!F]>
  return %cast : !trait.claim<@FoldFn[i32, i64, !trait.proj<@Fold[!F], "Item">] by @FoldFn_i32_alt>
}
