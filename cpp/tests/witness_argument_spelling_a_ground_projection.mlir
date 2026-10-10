// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWER
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=KEEP

// A citation's argument may spell a ground projection -- here the one @S_i64's
// where-equality determines, read off the refl premise the witness supplies for
// that entry, still unresolved. The witness keeps verifying through
// monomorphization, and @read lowers to its identity. @keep returns its witness,
// so the witness survives instantiation with its premise and the sweeps reach
// both.

// CHECK-LABEL: func.func @read
// CHECK: %[[U:.*]] = trait.witness refl : !trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.proj<@Marker[i64], "M">>
// CHECK: by @S_i64[!trait.proj<@Marker[i64], "M">] given(%[[U]])
// LOWER-LABEL: func.func @read(%arg0: i1) -> i1
// LOWER-NEXT: return %arg0 : i1
!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Marker(%self: !trait.claim<@Marker[!S]>) {
  trait.assoc_type @M
}
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) {
  trait.assoc_type @M = i1
}
trait.trait private @S(%self: !trait.claim<@S[!S]>) {
  trait.assoc_type @Out
}
trait.impl private @S_i64(%self: !trait.claim<@S[i64]>, %m: !trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.poly<0>>) {
  trait.assoc_type @Out = !trait.poly<0>
}
// The argument for !U is the projection the where-equality spells, still spelled.
func.func @read(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  %u = trait.witness refl : !trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.proj<@Marker[i64], "M">>
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves !trait.proj<@Marker[i64], "M"> by @S_i64[!trait.proj<@Marker[i64], "M">]
    given(%u) : (!trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.proj<@Marker[i64], "M">>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to !trait.proj<@Marker[i64], "M"> via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">>)
  %m = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64
    : !trait.claim<!trait.proj<@Marker[i64], "M"> = i1>
  %b = trait.coerce %r : !trait.proj<@Marker[i64], "M"> to i1 via (%m)
    : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i1>)
  return %b : i1
}

// KEEP-LABEL: func.func @keep
// KEEP: %[[U:.*]] = trait.witness refl : !trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.proj<@Marker[i64], "M">>
// KEEP: by @S_i64[!trait.proj<@Marker[i64], "M">] given(%[[U]])
func.func @keep() -> !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">> {
  %u = trait.witness refl : !trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.proj<@Marker[i64], "M">>
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves !trait.proj<@Marker[i64], "M"> by @S_i64[!trait.proj<@Marker[i64], "M">]
    given(%u) : (!trait.claim<!trait.proj<@Marker[i64], "M"> = !trait.proj<@Marker[i64], "M">>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">>
  return %e : !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">>
}
