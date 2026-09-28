// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// The derived application is the impl's header at the stated arguments.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr[!T] {}
trait.impl private @Tr_tuple for @Tr[tuple<!U>] where [@Tr[!U]] {}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{impl '@Tr_tuple' at its stated arguments is an impl of #trait<application@Tr[tuple<!trait.poly<0>>]>, not of #trait<application@Tr[tuple<i64>]>}}
  %d = trait.derive @Tr[tuple<i64>] from @Tr_tuple[!U = !T] given(%t) : (!trait.claim<@Tr[!T]>)
  return
}

// -----

// One premise per where-clause entry, the equality entry included.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr[!T] { trait.assoc_type @Out }
trait.impl private @Tr_tuple for @Tr[tuple<!U>] where [@Tr[!U], !trait.proj<@Tr[!U], "Out"> = i64] {
  trait.assoc_type @Out = i64
}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{impl '@Tr_tuple' states 2 where-clause entries, and the derive supplies 1 premises}}
  %d = trait.derive @Tr[tuple<!T>] from @Tr_tuple[!U = !T] given(%t) : (!trait.claim<@Tr[!T]>)
  return
}

// -----

// Each premise is its entry at the stated arguments.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr[!T] { trait.assoc_type @Out }
trait.impl private @Tr_tuple for @Tr[tuple<!U>] where [@Tr[!U], !trait.proj<@Tr[!U], "Out"> = i64] {
  trait.assoc_type @Out = i64
}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{premise 1 is '!trait.claim<!trait.proj<@Tr[!trait.poly<0>], "Out"> = i64>', and the derive supplies '!trait.claim<@Tr[!trait.poly<0>]>'}}
  %d = trait.derive @Tr[tuple<!T>] from @Tr_tuple[!U = !T] given(%t, %t) : (!trait.claim<@Tr[!T]>, !trait.claim<@Tr[!T]>)
  return
}

// -----

// The stated arguments are the impl's own parameters, each once.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr[!T] {}
trait.impl private @Tr_tuple for @Tr[tuple<!U>] where [@Tr[!U]] {}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{the citation binds no argument for type parameter '!trait.poly<1>' of impl '@Tr_tuple'}}
  %d = trait.derive @Tr[tuple<!T>] from @Tr_tuple[] given(%t) : (!trait.claim<@Tr[!T]>)
  return
}
