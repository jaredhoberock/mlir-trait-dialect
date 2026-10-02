// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// A premise is read through the impls that resolve what it spells. The cited
// impl's second where entry is @CF[proj<@HG[i64],"Sub">], which resolves to
// @CF[i32] through @HG_i64, while the witness supplies @CF[i8]; i8 is not i32,
// so the premise is not supplied and the witness is refused.
//
// RUN: mlir-opt %s -verify-diagnostics

!G = !trait.poly<0>
trait.trait private @HG(%self: !trait.claim<@HG[!G]>) {
  trait.assoc_type @Sub
}
trait.trait private @CF(%self: !trait.claim<@CF[!G]>) {}
trait.trait private @Fn(%self: !trait.claim<@Fn[!G]>) {
  trait.assoc_type @Out
}
trait.impl private @HG_i64(%self: !trait.claim<@HG[i64]>) {
  trait.assoc_type @Sub = i32
}
trait.impl private @CF_i8(%self: !trait.claim<@CF[i8]>) {}
trait.impl private @Fn_impl(%self: !trait.claim<@Fn[!G]>, %hg: !trait.claim<@HG[!G]>, %cf: !trait.claim<@CF[!trait.proj<@HG[!G], "Sub">]>) {
  trait.assoc_type @Out = i1
}
func.func @f(%v: !trait.proj<@Fn[i64], "Out">, %hg: !trait.claim<@HG[i64] by @HG_i64>, %cf: !trait.claim<@CF[i8] by @CF_i8>) -> i1 {
  // expected-error@+1 {{premise 1 of impl '@Fn_impl' is '!trait.claim<@CF[!trait.proj<@HG[i64], "Sub">]>', and the witness supplies '!trait.claim<@CF[i8]>'}}
  %eq = trait.witness proj_resolve !trait.proj<@Fn[i64], "Out"> resolves i1 by @Fn_impl given(%hg, %cf)
    : (!trait.claim<@HG[i64] by @HG_i64>, !trait.claim<@CF[i8] by @CF_i8>)
    : !trait.claim<!trait.proj<@Fn[i64], "Out"> = i1>
  %c = trait.coerce %v : !trait.proj<@Fn[i64], "Out"> to i1 via (%eq)
    : (!trait.claim<!trait.proj<@Fn[i64], "Out"> = i1>)
  return %c : i1
}
