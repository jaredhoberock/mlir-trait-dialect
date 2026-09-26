// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// The obligation discharge in a projection-resolution witness compares an
// obligation and a premise modulo ground-projection resolution: the cited
// impl assumes @CF[proj<@HG[i64],"Sub">], the premise supplies @CF[i32], and
// @HG_i64 resolves the projection to i32, so the discharge succeeds. Two
// spellings of one ground application discharge each other; only a difference
// that survives resolution is a mismatch.
//
// RUN: mlir-opt %s | FileCheck %s

!G = !trait.poly<0>
trait.trait private @HG[!G] {
  trait.assoc_type @Sub
}
trait.trait private @CF[!G] {}
trait.trait private @Fn[!G] {
  trait.assoc_type @Out
}
trait.impl private @HG_i64 for @HG[i64] {
  trait.assoc_type @Sub = i32
}
trait.impl private @CF_i32 for @CF[i32] {}
trait.impl private @Fn_impl for @Fn[!G] where [@HG[!G], @CF[!trait.proj<@HG[!G], "Sub">]] {
  trait.assoc_type @Out = i1
}
func.func @f(%v: !trait.proj<@Fn[i64], "Out">, %hg: !trait.claim<@HG[i64] by @HG_i64>, %cf: !trait.claim<@CF[i32] by @CF_i32>) -> i1 {
  %eq = trait.witness proj_resolve !trait.proj<@Fn[i64], "Out"> resolves i1 by @Fn_impl[!G = i64] given(%hg, %cf)
    : (!trait.claim<@HG[i64] by @HG_i64>, !trait.claim<@CF[i32] by @CF_i32>)
    : !trait.claim<!trait.proj<@Fn[i64], "Out"> = i1>
  %c = trait.coerce %v : !trait.proj<@Fn[i64], "Out"> to i1 via (%eq)
    : (!trait.claim<!trait.proj<@Fn[i64], "Out"> = i1>)
  return %c : i1
}

// CHECK: trait.witness proj_resolve
