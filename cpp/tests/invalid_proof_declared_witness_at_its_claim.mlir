// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// A generic impl's method meets the trait's spelling @S[i64]::Out by an
// equality it alleges at the impl's parameter, which only an instance can
// decide. The proof's claim is such an instance: at T = i64 the allegation
// reads i64 = @S[i64]::Out, which @S_i64 resolves to i1, so the instance is
// refused.

!S = !trait.poly<0>
!U = !trait.poly<1>
!T = !trait.poly<2>

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
trait.impl private @Foo_T(%self: !trait.claim<@Foo[!trait.poly<0>]>) {
  trait.method @f(%x: !trait.poly<0>) -> !trait.proj<@S[i64], "Out"> {
    // expected-error @below {{alleges 'i64' = '!trait.proj<@S[i64], "Out">', and impl selection resolves its sides to 'i64' and 'i1'}}
    // expected-error @below {{unproven monomorphic claim '!trait.claim<i64 = !trait.proj<@S[i64], "Out">>' after instantiate-monomorphs}}
    %e = trait.allege !trait.poly<0> = !trait.proj<@S[i64], "Out">
    %c = trait.coerce %x : !trait.poly<0> to !trait.proj<@S[i64], "Out"> via (%e) : (!trait.claim<!trait.poly<0> = !trait.proj<@S[i64], "Out">>)
    trait.return %c : !trait.proj<@S[i64], "Out">
  }
}
trait.proof private @Foo_i64_p {
  %d = trait.derive @Foo[i64] from @Foo_T given()
  trait.return %d : !trait.claim<@Foo[i64]>
}
func.func @main(%x: i64) -> i1 {
  %w = trait.witness @Foo_i64_p for @Foo[i64]
  %r = trait.method.call %w @Foo[i64]::@f(%x) : (i64) -> !trait.proj<@S[i64], "Out"> by @Foo_i64_p
  %m = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64 : !trait.claim<!trait.proj<@Marker[i64], "M"> = i1>
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64 given(%m) : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i1>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %c = trait.coerce %r : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %c : i1
}
