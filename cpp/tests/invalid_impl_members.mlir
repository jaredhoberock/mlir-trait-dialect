// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// An impl pairs each member of its trait once with the function implementing
// it, which leads with the impl's block arguments, or with the member itself
// where the trait gives it a default; and binds each associated type of its
// trait, over its own parameters and the associated type's own.

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
// expected-error @below {{pairs no implementation with member @T.m of trait '@T'}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) {
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
// expected-error @below {{pairs @U.m, which is no member of trait '@T'}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@U.m -> @I.m] {
}
func.func private @I.m(%self: !trait.claim<@T[i32]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
// expected-error @below {{pairs member @T.m twice}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@T.m -> @I.m, @T.m -> @I.m] {
}
func.func private @I.m(%self: !trait.claim<@T[i32]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
// expected-error @below {{leaves member @T.m to its trait, which gives it no default}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@T.m -> @T.m] {
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
// expected-error @below {{implements member @T.m by @I.m, which is no function with a body}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) members [@T.m -> @I.m] {
}
func.func private @I.m(!trait.claim<@T[i32]>) -> i64

// -----

!S = !trait.poly<0>
trait.trait private @B(!trait.claim<@B[!S]>)
trait.trait private @T(!trait.claim<@T[!S]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>) -> i64
// expected-error @below {{implements member @T.m by @I.m, whose leading parameters are not the impl's block arguments}}
trait.impl private @I(%self: !trait.claim<@T[tuple<!S>]>, %b: !trait.claim<@B[!S]>) members [@T.m -> @I.m] {
}
func.func private @I.m(%self: !trait.claim<@T[tuple<!S>]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>)
// expected-error @below {{pairs members as [@T.m], which is no pair of a member and the function implementing it}}
"trait.impl"() <{sym_name = "I", bindings = {}, members = [[@T.m]]}> ({
^bb0(%self: !trait.claim<@T[i32]>):
  trait.return
}) : () -> ()

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) assoc {A = 0}
// expected-error @below {{missing binding for associated type 'A' of trait '@T'}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) {
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>)
// expected-error @below {{binds associated type 'A', which trait '@T' does not declare}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) bindings {A = i64} {
}

// -----

!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>) assoc {A = 1}
// The impl takes one parameter, label 0, and the associated type one of its
// own, label 1: label 2 stands for nothing.
// expected-error @below {{binds associated type 'A' to a type mentioning type parameter '!trait.poly<2>', which neither impl '@I' nor the associated type declares}}
trait.impl private @I(%self: !trait.claim<@T[tuple<!S>]>) bindings {A = tuple<!S, !trait.poly<1>, !trait.poly<2>>} {
}

// -----

// A symbol nested in an impl is refused: an impl contains evidence alone, and is
// no symbol table.
!S = !trait.poly<0>
trait.trait private @T(!trait.claim<@T[!S]>)
// expected-error @below {{'trait.impl' op unexpected child op 'func.func'}}
trait.impl private @I(%self: !trait.claim<@T[i32]>) {
  func.func private @f() {
    return
  }
}
