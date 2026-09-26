// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWER
// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=KEEP

// An argument may spell a ground projection -- here the one @S_i64's
// where-equality spells, still unresolved. The witness is sealed whole under
// every rewrite, so no sweep resolving ground projections rewrites the argument
// out from under the endpoints; the witness keeps verifying through
// monomorphization, and @read lowers to its identity. @keep returns its witness,
// so the witness survives instantiation and the sweeps reach it.

// CHECK-LABEL: func.func @read
// CHECK: by @S_i64[!trait.poly<1> = !trait.proj<@Marker[i64], "M">]
// LOWER-LABEL: func.func @read(%arg0: i1) -> i1
// LOWER-NEXT: return %arg0 : i1
!S = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Marker[!S] {
  trait.assoc_type @M
}
trait.impl private @Marker_i64 for @Marker[i64] {
  trait.assoc_type @M = i1
}
trait.trait private @S[!S] {
  trait.assoc_type @Out
}
trait.impl private @S_i64 for @S[i64] where [!trait.proj<@Marker[i64], "M"> = !U]
    witnesses [#trait<witness !trait.proj<@Marker[i64], "M"> = i1 by @Marker_i64>] {
  trait.assoc_type @Out = !U
}
// The argument for !U is the projection the where-equality spells, still spelled.
func.func @read(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves !trait.proj<@Marker[i64], "M"> by @S_i64[!U = !trait.proj<@Marker[i64], "M">]
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
// KEEP: by @S_i64[!trait.poly<1> = !trait.proj<@Marker[i64], "M">]
func.func @keep() -> !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">> {
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves !trait.proj<@Marker[i64], "M"> by @S_i64[!U = !trait.proj<@Marker[i64], "M">]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">>
  return %e : !trait.claim<!trait.proj<@S[i64], "Out"> = !trait.proj<@Marker[i64], "M">>
}
