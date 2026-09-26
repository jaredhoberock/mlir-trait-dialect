// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// A template's projection-resolution witness leaves a premise of its cited impl
// that still spells a type variable to the instances. Each monomorphic clone is
// one, and decides it there -- before any rewrite can fold the witness away --
// through the evidence the instance holds (the clone's proven premises, and the
// cited impl's declaration witnesses, each rechecked where it is carried) and
// what the stage's impl selection has settled.

// @S_blanket applies where Tensor[T]::Shape = i64; at T = i8 the premise the
// clone carries, proved by @Tensor_i8, binds Shape to tuple<i64, i64>.

!T = !trait.poly<0>

trait.trait private @Tensor[!T] { trait.assoc_type @Shape }
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = tuple<i64, i64>
}
trait.trait private @S[!T] { trait.assoc_type @Out }
trait.impl private @S_blanket for @S[!T]
    where [@Tensor[!T], !trait.proj<@Tensor[!T], "Shape"> = i64] {
  trait.assoc_type @Out = i64
}

func.func private @f(%t: !trait.claim<@Tensor[!T]>) -> i64 {
  %v = ub.poison : !trait.proj<@S[!T], "Out">
  // expected-error @below {{impl '@S_blanket' applies where '!trait.proj<@Tensor[!trait.poly<0>], "Shape">' = 'i64', and nothing here makes 'tuple<i64, i64>' and 'i64' one type at '!trait.claim<@S[i8]>'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@S[i8], "Out"> = i64>' after instantiate-monomorphs}}
  %e = trait.witness proj_resolve !trait.proj<@S[!T], "Out"> resolves i64 by @S_blanket[!T = !T] given(%t)
    : (!trait.claim<@Tensor[!T]>) : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
  %r = trait.coerce %v : !trait.proj<@S[!T], "Out"> to i64 via (%e)
    : (!trait.claim<!trait.proj<@S[!T], "Out"> = i64>)
  return %r : i64
}

func.func @main() {
  %t = trait.allege @Tensor[i8]
  %r = trait.func.call @f(%t) : (!trait.claim<@Tensor[i8]>) -> i64
  return
}

// -----

// A declaration witness citing a conditional impl holds only where that impl
// does: at T = i8, @A_blanket's own premise Tensor[i8]::Shape = i64 is false,
// so @S_blanket's witness contributes no rule for A[i8]::Out. Before the stage
// has selected an impl for A[i8] the premise stands unsettled; once it has,
// A[i8]::Out reads as @A_i8's i32. Either way the instance is refused and the
// call is left standing.

!T = !trait.poly<0>
!U = !trait.poly<1>

trait.trait private @Tensor[!T] {
  trait.assoc_type @Shape
}
trait.impl private @Tensor_i8 for @Tensor[i8] {
  trait.assoc_type @Shape = tuple<i64, i64>
}

trait.trait private @A[!T] {
  trait.assoc_type @Out
}
trait.impl private @A_blanket for @A[!T]
    where [@Tensor[!T], !trait.proj<@Tensor[!T], "Shape"> = i64] {
  trait.assoc_type @Out = i64
}
trait.impl private @A_i8 for @A[i8] {
  trait.assoc_type @Out = i32
}

trait.trait private @S[!T] {
  trait.assoc_type @Out
}
trait.impl private @S_blanket for @S[!T]
    where [@Tensor[!T], !trait.proj<@A[!T], "Out"> = !U]
    witnesses [#trait<witness !trait.proj<@A[!T], "Out"> = i64 by @A_blanket[!T = !T]>] {
  trait.assoc_type @Out = !U
}

func.func private @f(%t: !trait.claim<@Tensor[!T]>) {
  // expected-error @below {{impl '@S_blanket' applies where '!trait.proj<@A[!trait.poly<0>], "Out">' = '!trait.poly<1>', and nothing here settles '!trait.proj<@A[i8], "Out">' = 'i64' at '!trait.claim<@S[i8]>'}}
  // expected-error @below {{impl '@S_blanket' applies where '!trait.proj<@A[!trait.poly<0>], "Out">' = '!trait.poly<1>', and nothing here makes 'i32' and 'i64' one type at '!trait.claim<@S[i8]>'}}
  %e = trait.witness proj_resolve !trait.proj<@S[!T], "Out"> resolves i64 by @S_blanket[!T = !T, !U = i64] given(%t)
    : (!trait.claim<@Tensor[!T]>) : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
  return
}

func.func @main() {
  %t = trait.allege @Tensor[i8]
  // expected-error @below {{'trait.func.call' op rewritable generic call survived instantiate-monomorphs}}
  trait.func.call @f(%t) : (!trait.claim<@Tensor[i8]>) -> ()
  return
}

// -----

// The template reads @S_i64's premise as i1 = T; the instance at T = i64 makes
// it i1 = i64.

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
// Template: the where-equality reads i1 = !T, deferred to the instances.
func.func private @gen(%v: !trait.proj<@S[i64], "Out">, %x: !T) -> !T {
  // expected-error @below {{impl '@S_i64' applies where '!trait.proj<@Marker[i64], "M">' = '!trait.poly<1>', and nothing here makes 'i1' and 'i64' one type at '!trait.claim<@S[i64]>'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@S[i64], "Out"> = i64>' after instantiate-monomorphs}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves !T by @S_i64[!U = !T]
    : !trait.claim<!trait.proj<@S[i64], "Out"> = !T>
  %r = trait.coerce %v : !trait.proj<@S[i64], "Out"> to !T via (%e)
    : (!trait.claim<!trait.proj<@S[i64], "Out"> = !T>)
  return %r : !T
}
// Instance at T := i64 -- WRONG: @S[i64]::Out is i1.
func.func @caller(%v: !trait.proj<@S[i64], "Out">) -> i64 {
  %x = arith.constant 7 : i64
  %r = trait.func.call @gen(%v, %x) : (!trait.proj<@S[i64], "Out">, i64) -> i64
  return %r : i64
}
