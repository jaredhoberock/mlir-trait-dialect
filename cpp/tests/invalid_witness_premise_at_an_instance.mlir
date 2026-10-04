// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics -pass-pipeline='builtin.module(monomorphize-trait)'

// A template's projection-resolution witness supplies its cited impl's premises
// as values, and a premise the template can only allege still spells a type
// variable. Each monomorphic clone decides it there through what the stage's
// impl selection settles: the allegation is an obligation outstanding, so it
// stands to be decided whatever becomes of the witness citing it.

// @S_blanket applies where Tensor[T]::Shape = i64; at T = i8 @Tensor_i8 binds
// Shape to tuple<i64, i64>, so the clone's allegation is false -- and no impl
// serves @S[i8], so the clone's @S[i8]::Out denotes nothing: selection refuses
// it, naming @S_blanket's false premise, and it stays spelled.

!T = !trait.poly<0>

trait.trait private @Tensor(%self: !trait.claim<@Tensor[!T]>) { trait.assoc_type @Shape }
trait.impl private @Tensor_i8(%self: !trait.claim<@Tensor[i8]>) {
  trait.assoc_type @Shape = tuple<i64, i64>
}
trait.trait private @S(%self: !trait.claim<@S[!T]>) { trait.assoc_type @Out }
// expected-note @+1 {{unsatisfiable candidate}}
trait.impl private @S_blanket(%self: !trait.claim<@S[!T]>, %tensor: !trait.claim<@Tensor[!T]>, %shape: !trait.claim<!trait.proj<@Tensor[!T], "Shape"> = i64>) {
  trait.assoc_type @Out = i64
}

func.func private @f(%t: !trait.claim<@Tensor[!T]>) -> i64 {
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@S[i8], "Out">'}}
  // expected-error @below {{unresolved projection '!trait.proj<@S[i8], "Out">' after instantiate-monomorphs}}
  %v = ub.poison : !trait.proj<@S[!T], "Out">
  // expected-error @below {{alleges '!trait.proj<@Tensor[i8], "Shape">' = 'i64', and impl selection resolves its sides to 'tuple<i64, i64>' and 'i64'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Tensor[i8], "Shape"> = i64>' after instantiate-monomorphs}}
  %s = trait.allege !trait.proj<@Tensor[!T], "Shape"> = i64
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@S[i8], "Out"> = i64>' after instantiate-monomorphs}}
  %e = trait.witness proj_resolve !trait.proj<@S[!T], "Out"> resolves i64 by @S_blanket given(%t, %s)
    : (!trait.claim<@Tensor[!T]>, !trait.claim<!trait.proj<@Tensor[!T], "Shape"> = i64>) : !trait.claim<!trait.proj<@S[!T], "Out"> = i64>
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

// The template alleges @S_i64's premise as i1 = T; the instance at T = i64
// makes it i1 = i64.

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
trait.impl private @S_i64(%self: !trait.claim<@S[i64]>, %m: !trait.claim<!trait.proj<@Marker[i64], "M"> = !U>) {
  trait.assoc_type @Out = !U
}
func.func private @gen(%v: !trait.proj<@S[i64], "Out">, %x: !T) -> !T {
  // expected-error @below {{alleges '!trait.proj<@Marker[i64], "M">' = 'i64', and impl selection resolves its sides to 'i1' and 'i64'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@Marker[i64], "M"> = i64>' after instantiate-monomorphs}}
  %m = trait.allege !trait.proj<@Marker[i64], "M"> = !T
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@S[i64], "Out"> = i64>' after instantiate-monomorphs}}
  %e = trait.witness proj_resolve !trait.proj<@S[i64], "Out"> resolves !T by @S_i64 given(%m)
    : (!trait.claim<!trait.proj<@Marker[i64], "M"> = !T>)
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
