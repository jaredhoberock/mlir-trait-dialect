// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A projection-resolution witness carries the cited impl's substitution, one
// argument keyed by each of the impl's own parameters, and the verifier holds
// the impl to it: its keys, the binding it makes, and the where-clause
// equalities it must satisfy.

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

func.func @missing_parameter(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  // expected-error @below {{the citation binds no argument for type parameter '!trait.poly<1>' of impl '@S_i64'}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %r : i1
}

// -----

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

func.func @key_is_not_a_parameter_of_the_impl(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  // expected-error @below {{the citation binds '!trait.poly<7>', which is not a type parameter of impl '@S_i64'}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64[!trait.poly<7> = i1]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %r : i1
}

// -----

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

func.func @duplicate_key(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  // expected-error @below {{the citation binds type parameter '!trait.poly<1>' of impl '@S_i64' twice}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64[!U = i1, !U = i1]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %r : i1
}

// -----

// expected-error @below {{a type binding's key must be a type parameter, found 'i64'}}
func.func private @key_is_not_a_type_parameter() attributes {evidence = #trait<witness !trait.proj<@A[i64], "Item"> = i1 by @A_impl[i64 = i1]>}

// -----

// The argument makes the binding the certified resolution, but the impl applies
// only where @Marker[i64]::M is its argument, and @Marker[i64]::M is i1.

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

func.func @contradicts_the_where_clause(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  // expected-error @below {{impl '@S_i64' applies where '!trait.proj<@Marker[i64], "M">' = '!trait.poly<1>', and nothing here makes 'i1' and 'i64' one type at '!trait.claim<@S[i64]>'}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i64 by @S_i64[!U = i64]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i64>)
  return %r : i64
}

// -----

// The same contradiction in a declaration-level witness, which the citing impl's
// verification refuses.

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

// expected-error @below {{impl '@S_i64' applies where '!trait.proj<@Marker[i64], "M">' = '!trait.poly<1>', and nothing here makes 'i1' and 'i64' one type at '!trait.claim<@S[i64]>'}}
trait.impl private @Foo_i64 for @Foo[i64]
    witnesses [#trait<witness !trait.proj<@S[i64], "Out"> = i64 by @S_i64[!U = i64]>] {
  func.func @f(%x: i64) -> i64 {
    return %x : i64
  }
}

// -----

// The argument satisfies the where clause, but the binding it makes is not the
// certified resolution.

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

func.func @binding_disagrees(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  // expected-error @below {{impl '@S_i64' binds the projection to 'i1', not the certified resolution 'i64'}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i64 by @S_i64[!U = i1]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i64>)
  return %r : i64
}

// -----

// The arguments must make the impl's header the projection's application.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A[!S] {
  trait.assoc_type @Item
}
trait.impl private @A_tuple for @A[tuple<!U>] {
  trait.assoc_type @Item = !U
}

func.func @header_disagrees(%v: !trait.proj<@A[tuple<i64>], "Item">) -> i32 {
  // expected-error @below {{impl '@A_tuple' at the witness's arguments is an impl for '!trait.claim<@A[tuple<i32>]>', not for the projection's application '!trait.claim<@A[tuple<i64>]>'}}
  %e = trait.witness proj_resolve !trait.proj<@A[tuple<i64>], "Item"> resolves i32 by @A_tuple[!U = i32]
    : !trait.claim<!trait.proj<@A[tuple<i64>], "Item"> = i32>
  %r = trait.coerce %v : !trait.proj<@A[tuple<i64>], "Item"> to i32 via (%e)
    : (!trait.claim<!trait.proj<@A[tuple<i64>], "Item"> = i32>)
  return %r : i32
}

// -----

!S = !trait.poly<0>

trait.trait private @X[!S] {}
trait.impl private @X_i64 for @X[i64] {}

// expected-error @below {{an application witness names its impl alone; the impl is read at the application it discharges}}
func.func private @application_arm_with_arguments() attributes {evidence = #trait<witness @X[i64] by @X_i64[!S = i64]>}
