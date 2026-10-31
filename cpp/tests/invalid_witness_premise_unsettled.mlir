// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// A closed premise of the cited impl -- no type variable left -- is settled by
// the evidence the citation supplies for it. @S_gen's head is generic and its
// premise Marker[T]::M = U carries the argument U, so a citation alleging
// Marker[i64]::M = i64 makes the binding i64; the allegation is decided through
// the module's impls, and @Marker_i64's i1 contradicts it.

!S = !trait.poly<0>
!U = !trait.poly<1>
!T = !trait.poly<2>

trait.trait private @Marker(!trait.claim<@Marker[!S]>) assoc {M = 0}
trait.impl private @Marker_i64(%self: !trait.claim<@Marker[i64]>) bindings {M = i1} {
}
trait.trait private @S(!trait.claim<@S[!S]>) assoc {Out = 0}
trait.impl private @S_gen(%self: !trait.claim<@S[!trait.poly<0>]>, %m: !trait.claim<!trait.proj<@Marker[!trait.poly<0>], "M"> = !trait.poly<1>>) bindings {Out = !U} {
}

// WRONG: @Marker[i64]::M is i1, so @S[i64]::Out is i1; the citation says i64.
func.func @wrong(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  // expected-error @below {{alleges '!trait.proj<@Marker[i64], "M">' = 'i64', and impl selection resolves its sides to 'i1' and 'i64'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Marker[i64], "M"> = i64>' after instantiate-monomorphs}}
  %m = trait.allege !trait.proj<@Marker[i64], "M"> = i64
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves i64 by @S_gen[i64, i64] given(%m)
    : (!trait.claim<!trait.proj<@Marker[i64], "M"> = i64>)
    : !trait.claim<!trait.proj<@S[i64], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = i64>)
  return %r : i64
}
