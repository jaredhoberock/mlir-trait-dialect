// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s --check-prefix=LOWER

// @S_i64's parameter !U appears nowhere in its header: only its where-clause
// equality determines it, so no reading of the projection @S[i64]::Out can say
// what it is. A witness citing @S_i64 supplies one claim for that entry, by
// position -- here @Marker[i64]::M = i1, itself a witness through @Marker_i64 --
// and the verifier reads the argument off it: !U is i1. @Foo_i64's method keeps
// the trait's spelling @S[i64]::Out and coerces the i1 it computes across that
// resolution; the use site coerces a value across the same resolution.

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

trait.trait private @Foo(%self: !trait.claim<@Foo[!S]>) {
  trait.method @f(!S) -> !trait.proj<@S[i64], "Out">
}

// CHECK: trait.impl private @Foo_i64(%self: !trait.claim<@Foo[i64]>)
// CHECK: %[[M:.*]] = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64
// CHECK: trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64 given(%[[M]])
trait.impl private @Foo_i64(%self: !trait.claim<@Foo[i64]>) {
  trait.method @f(%x: i64) -> !trait.proj<@S[i64], "Out"> {
    %t = arith.constant true
    %m = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64
      : !trait.claim<!trait.proj<@Marker[i64], "M"> = i1>
    %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64
      given(%m) : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i1>)
      : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
    %r = trait.coerce %t : i1 to !trait.proj<@S[i64], "Out"> via (%e)
      : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
    trait.return %r : !trait.proj<@S[i64], "Out">
  }
}

// CHECK-LABEL: func.func @read
// CHECK: %[[M2:.*]] = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64
// CHECK: trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64 given(%[[M2]])
// LOWER-LABEL: func.func @read(%arg0: i1) -> i1
// LOWER-NEXT: return %arg0 : i1
func.func @read(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  %m = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64
    : !trait.claim<!trait.proj<@Marker[i64], "M"> = i1>
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64
    given(%m) : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i1>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %r : i1
}
