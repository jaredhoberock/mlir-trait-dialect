// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A closed premise of the cited impl -- no type variable left -- must be
// settled by the citation's evidence, or rest on one of the cited impl's own
// premises. @S_gen's head is generic, so it carries no declaration witness for
// Marker[i64]::M, and it states no premise Marker[T]. A use reads such a
// projection through the module's impls, as its head comparison does, and the
// argument i64 contradicts @Marker_i64's i1.

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
trait.impl private @S_gen for @S[!T] where [!trait.proj<@Marker[!T], "M"> = !U] {
  trait.assoc_type @Out = !U
}

// WRONG: @Marker[i64]::M is i1, so @S[i64]::Out is i1; the witness says i64 with U := i64.
func.func @wrong(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  // expected-error @below {{impl '@S_gen' applies where '!trait.proj<@Marker[!trait.poly<2>], "M">' = '!trait.poly<1>', and nothing here makes 'i1' and 'i64' one type at '!trait.claim<@S[i64]>'}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i64 by @S_gen[!T = i64, !U = i64]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i64>)
  return %r : i64
}

// -----

// The same citation as a declaration witness. An impl's verification reads no
// module, so the premise stands unsettled and the citing impl is refused.

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
trait.impl private @S_gen for @S[!T] where [!trait.proj<@Marker[!T], "M"> = !U] {
  trait.assoc_type @Out = !U
}
trait.trait private @Foo[!S] {
  func.func private @f(!S) -> !trait.proj<@S[i64], "Out">
}
// WRONG declaration witness: S[i64]::Out is i1 through the module, the witness says i64.
// expected-error @below {{impl '@S_gen' applies where '!trait.proj<@Marker[!trait.poly<2>], "M">' = '!trait.poly<1>', and nothing here settles '!trait.proj<@Marker[i64], "M">' = 'i64' at '!trait.claim<@S[i64]>'}}
trait.impl private @Foo_i64 for @Foo[i64]
    witnesses [#trait<witness !trait.proj<@S[i64], "Out"> = i64 by @S_gen[!T = i64, !U = i64]>] {
  func.func @f(%x: i64) -> i64 {
    return %x : i64
  }
}
func.func @main(%x: i64) -> i1 {
  %w = trait.allege @Foo[i64]
  %r = trait.method.call %w @Foo[i64]::@f(%x) : (i64) -> !trait.proj<@S[i64], "Out">
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_gen[!T = i64, !U = i1]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %c = trait.coerce %r : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %c : i1
}
