// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A bound requirement is selected at one argument per parameter it binds.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>) {
  // expected-error @below {{requirement 0 binds 1 parameters, and 0 arguments are supplied}}
  %m = trait.project %h[0] : !trait.claim<@Has[!trait.poly<2>]> -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>
  return
}

// -----

// A requirement that binds nothing takes no arguments.

!S = !trait.poly<0>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [@Marker[!S]] {}
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>) {
  // expected-error @below {{requirement 0 binds no parameters, and 1 arguments are supplied}}
  %m = trait.project %h[0] for [i1] : !trait.claim<@Has[!trait.poly<2>]> -> !trait.claim<@Marker[!trait.poly<2>]>
  return
}

// -----

// A bound requirement's premise travels with the hop.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>) {
  // expected-error @below {{requirement 0 states 1 premises, and the hop supplies 0}}
  %m = trait.project %h[0] for [i1] : !trait.claim<@Has[!trait.poly<2>]> -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>
  return
}

// -----

// The premise claim is the premise at the hop's arguments.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] where [@Marker[!X]] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>, %p: !trait.claim<@Marker[i64]>) {
  // expected-error @below {{premise 0 of requirement 0 is '!trait.claim<@Marker[i1]>', and the hop supplies '!trait.claim<@Marker[i64]>'}}
  %m = trait.project %h[0] for [i1] given(%p : !trait.claim<@Marker[i64]>) : !trait.claim<@Has[!trait.poly<2>]> -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>
  return
}

// -----

// The result type spells the conclusion at the hop's arguments.

!S = !trait.poly<0>
!X = !trait.poly<1>
trait.trait private @Marker[!S] {}
trait.trait private @Has[!S] where [forall [!X] -> @Marker[!trait.proj<@Has[!S], "A", [!X]>]] {
  trait.assoc_type @A<[!X]>
}
func.func private @f(%h: !trait.claim<@Has[!trait.poly<2>]>) {
  // expected-error @below {{type mismatch: expected '!trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i1]>]>' but found '!trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i64]>]>'}}
  %m = trait.project %h[0] for [i1] : !trait.claim<@Has[!trait.poly<2>]> -> !trait.claim<@Marker[!trait.proj<@Has[!trait.poly<2>], "A", [i64]>]>
  return
}
