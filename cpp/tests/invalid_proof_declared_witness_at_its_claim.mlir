// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics

// A generic impl's declaration witness is verified at the impl's parameters,
// where a premise of its cited impl still spelling one of them is left to the
// instances. A proof's claim is such an instance: @Foo_T's witness, rebuilt at
// T = i64, makes @S_i64's premise i1 = i64, so the proof is refused.

!S = !trait.poly<0>
!U = !trait.poly<1>
!T = !trait.poly<2>

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
// Template impl over !T: its declaration witness reads i1 = !T, deferred.
trait.impl private @Foo_T for @Foo[!T]
    witnesses [#trait<witness !trait.proj<@S[i64], "Out"> = !T by @S_i64[!U = !T]>] {
  func.func @f(%x: !T) -> !T {
    return %x : !T
  }
}
// Instance at T := i64 -- WRONG: @S[i64]::Out is i1, but @Foo_T at i64 returns i64.
// expected-error @below {{impl '@S_i64' applies where '!trait.proj<@Marker[i64], "M">' = '!trait.poly<1>', and nothing here makes 'i1' and 'i64' one type at '!trait.claim<@S[i64]>'}}
trait.proof private @Foo_i64_p proves @Foo_T for @Foo[i64] given []
func.func @main(%x: i64) -> i1 {
  %w = trait.allege @Foo[i64]
  %r = trait.method.call %w @Foo[i64]::@f(%x) : (i64) -> !trait.proj<@S[i64], "Out">
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64[!U = i1]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %c = trait.coerce %r : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %c : i1
}
