// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A derive's impl arguments are read off its derived application and its
// premises; the derived application is the impl's header at those arguments.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) {}
trait.impl private @Tr_tuple(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>, %tr: !trait.claim<@Tr[!trait.poly<0>]>) {}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{impl '@Tr_tuple' at the arguments the citation gives it proves '!trait.claim<@Tr[tuple<!trait.poly<0>>]>', not '!trait.claim<@Tr[i64]>'}}
  %d = trait.derive @Tr[i64] from @Tr_tuple given(%t) : (!trait.claim<@Tr[!T]>)
  return
}

// -----

// One premise per where-clause entry, the equality entry included.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.assoc_type @Out }
trait.impl private @Tr_tuple(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>, %tr: !trait.claim<@Tr[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Tr[!trait.poly<0>], "Out"> = i64>) {
  trait.assoc_type @Out = i64
}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{impl '@Tr_tuple' has 2 where entries, and the citation supplies 1 claims}}
  %d = trait.derive @Tr[tuple<!T>] from @Tr_tuple given(%t) : (!trait.claim<@Tr[!T]>)
  return
}

// -----

// Each premise is its entry at the citation's arguments.

!T = !trait.poly<0>
!U = !trait.poly<1>
trait.trait private @Tr(%self: !trait.claim<@Tr[!T]>) { trait.assoc_type @Out }
trait.impl private @Tr_tuple(%self: !trait.claim<@Tr[tuple<!trait.poly<0>>]>, %tr: !trait.claim<@Tr[!trait.poly<0>]>, %out: !trait.claim<!trait.proj<@Tr[!trait.poly<0>], "Out"> = i64>) {
  trait.assoc_type @Out = i64
}
func.func private @g(%t: !trait.claim<@Tr[!T]>) {
  // expected-error @below {{premise 1 of impl '@Tr_tuple' is '!trait.claim<!trait.proj<@Tr[!trait.poly<0>], "Out"> = i64>', and the derive supplies '!trait.claim<@Tr[!trait.poly<0>]>'}}
  %d = trait.derive @Tr[tuple<!T>] from @Tr_tuple given(%t, %t) : (!trait.claim<@Tr[!T]>, !trait.claim<@Tr[!T]>)
  return
}
