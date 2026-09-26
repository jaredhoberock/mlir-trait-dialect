// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWER

// @S_i64's parameter !U appears nowhere in its header: only its where-clause
// equality determines it, so no reading of the projection @S[i64]::Out can say
// what it is. A witness citing @S_i64 carries it, keyed by the parameter --
// @S_i64[!U = i1] -- and the verifier substitutes it: the binding !U is i1, and
// the where-clause equality @Marker[i64]::M = i1 holds through @S_i64's own
// declaration witness. The declaration-level witness lets @Foo_i64's method
// return i1 where the trait declares @S[i64]::Out; the use-site witness coerces
// a value across the same resolution.

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

trait.trait private @Foo[!S] {
  func.func private @f(!S) -> !trait.proj<@S[i64], "Out">
}

// CHECK: trait.impl private @Foo_i64 for @Foo[i64]witnesses [#trait<witness!trait.proj<@S[i64], "Out"> = i1 by @S_i64[!trait.poly<1> = i1]>]
trait.impl private @Foo_i64 for @Foo[i64]
    witnesses [#trait<witness !trait.proj<@S[i64], "Out"> = i1 by @S_i64[!U = i1]>] {
  func.func @f(%x: i64) -> i1 {
    %t = arith.constant true
    return %t : i1
  }
}

// CHECK-LABEL: func.func @read
// CHECK: trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64[!trait.poly<1> = i1] : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
// LOWER-LABEL: func.func @read(%arg0: i1) -> i1
// LOWER-NEXT: return %arg0 : i1
func.func @read(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64[!U = i1]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %r : i1
}
