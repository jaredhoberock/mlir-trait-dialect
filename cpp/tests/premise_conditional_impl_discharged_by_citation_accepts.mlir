// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A proj_resolve witness cites a CONDITIONAL sibling impl whose where entry
// the citing impl's own where clause does NOT cover. @Sib_i64_cond binds
// Sib[i64]::Elem = i32 and takes @Y[i64]; @Host_i64 declares no where clause,
// so the witness supplies the entry itself, by position: a witness of @Y_i64,
// an unconditional impl of @Y[i64]. The method keeps the trait's spelling
// (i64) -> Sib[i64]::Elem, and its body coerces the i32 it computes through the
// witness. The premise is supplied by position; verification resolves the
// symbols it names and never scans the module for one.

!S = !trait.poly<0>

trait.trait private @Y(%self: !trait.claim<@Y[!S]>) {}

trait.impl private @Y_i64(%self: !trait.claim<@Y[i64]>) {}

trait.trait private @Sib(%self: !trait.claim<@Sib[!S]>) {
  trait.assoc_type @Elem
}

trait.impl private @Sib_i64_cond(%self: !trait.claim<@Sib[i64]>, %y: !trait.claim<@Y[i64]>) {
  trait.assoc_type @Elem = i32
}

trait.trait private @Host(%self: !trait.claim<@Host[!S]>) {
  trait.method @make(!S) -> !trait.proj<@Sib[!S], "Elem">
}

// CHECK: trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>)
// CHECK: trait.witness @Y_i64 for @Y[i64]
// CHECK: trait.witness proj_resolve !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond given(
// CHECK: trait.coerce
trait.impl private @Host_i64(%self: !trait.claim<@Host[i64]>) {
  trait.method @make(%x: i64) -> !trait.proj<@Sib[i64], "Elem"> {
    %r = ub.poison : i32
    %y = trait.witness @Y_i64 for @Y[i64]
    %e = trait.witness proj_resolve !trait.proj<@Sib[i64], "Elem"> resolves i32 by @Sib_i64_cond
      given(%y) : (!trait.claim<@Y[i64] by @Y_i64>)
      : !trait.claim<!trait.proj<@Sib[i64], "Elem"> = i32>
    %c = trait.coerce %r : i32 to !trait.proj<@Sib[i64], "Elem"> via (%e)
      : (!trait.claim<!trait.proj<@Sib[i64], "Elem"> = i32>)
    trait.return %c : !trait.proj<@Sib[i64], "Elem">
  }
}
