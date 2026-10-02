// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -split-input-file

// A projection-resolution witness reads the cited impl's arguments off the
// projection's application and the premises it supplies, one per where-clause
// entry, and the verifier holds the impl to them: the entries it states, the
// binding it makes, and the header it has at them.

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
trait.impl private @S_i64(%self: !trait.claim<@S[i64]>, %m: !trait.claim<!trait.proj<@Marker[i64], "M"> = !U>) {
  trait.assoc_type @Out = !U
}

func.func @missing_premise(%v: !trait.proj<@S[i64], "Out">) -> i1 {
  // expected-error @below {{impl '@S_i64' has 1 where entries, and the citation supplies 0 claims}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i1 by @S_i64
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i1>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i1 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i1>)
  return %r : i1
}

// -----

// The premise would make the binding the certified resolution, but the impl
// applies only where @Marker[i64]::M is its argument, and @Marker[i64]::M is i1:
// the premise's own evidence is refused.

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
trait.impl private @S_i64(%self: !trait.claim<@S[i64]>, %m: !trait.claim<!trait.proj<@Marker[i64], "M"> = !U>) {
  trait.assoc_type @Out = !U
}

func.func @contradicts_the_where_clause(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  // expected-error @below {{impl '@Marker_i64' binds the projection to 'i1', not the certified resolution 'i64'}}
  %m = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i64 by @Marker_i64
    : !trait.claim<!trait.proj<@Marker[i64], "M"> = i64>
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i64 by @S_i64 given(%m)
    : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i64>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i64>)
  return %r : i64
}

// -----

// The premise satisfies the where clause, but the binding it makes is not the
// certified resolution.

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
trait.impl private @S_i64(%self: !trait.claim<@S[i64]>, %m: !trait.claim<!trait.proj<@Marker[i64], "M"> = !U>) {
  trait.assoc_type @Out = !U
}

func.func @binding_disagrees(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  %m = trait.witness proj_resolve !trait.proj<@Marker[i64], "M"> resolves i1 by @Marker_i64
    : !trait.claim<!trait.proj<@Marker[i64], "M"> = i1>
  // expected-error @below {{impl '@S_i64' binds the projection to 'i1', not the certified resolution 'i64'}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i64 by @S_i64 given(%m)
    : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i1>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i64>)
  return %r : i64
}

// -----

// The projection's application fixes the impl's arguments through its header.

!S = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @A(%self: !trait.claim<@A[!S]>) {
  trait.assoc_type @Item
}
trait.impl private @A_tuple(%self: !trait.claim<@A[tuple<!U>]>) {
  trait.assoc_type @Item = !U
}

func.func @header_disagrees(%v: !trait.proj<@A[tuple<i64>], "Item">) -> i32 {
  // expected-error @below {{impl '@A_tuple' binds the projection to 'i64', not the certified resolution 'i32'}}
  %e = trait.witness proj_resolve !trait.proj<@A[tuple<i64>], "Item"> resolves i32 by @A_tuple
    : !trait.claim<!trait.proj<@A[tuple<i64>], "Item"> = i32>
  %r = trait.coerce %v : !trait.proj<@A[tuple<i64>], "Item"> to i32 via (%e)
    : (!trait.claim<!trait.proj<@A[tuple<i64>], "Item"> = i32>)
  return %r : i32
}
