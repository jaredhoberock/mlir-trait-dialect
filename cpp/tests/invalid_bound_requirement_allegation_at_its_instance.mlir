// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An impl alleges its bound requirement, and nothing proves the instance the
// use projects: the refusal at the use names the allegation.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Rule[!S] {
  trait.method @size(!S) -> i64
}
trait.trait private @Holds[!S] where [forall [!trait.bound<0>] -> @Rule[!trait.proj<@Holds[!S], "C", [!trait.bound<0>]>]] {
  trait.assoc_type @C<[!X]>
}
trait.impl private @Holds_i32 for @Holds[i32]
    witnesses [#trait<witness requirement 0 by allege @Rule[tuple<i64, i64>]>] {
  trait.assoc_type @C<[!X]> = tuple<i64, i64>
}

func.func private @use_rule(%m: !trait.claim<@Rule[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Rule[!M]::@size(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Holds[!T]>, %x: !trait.proj<@Holds[!T], "C", [i1]>) -> i64 {
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@Rule[tuple<i64, i64>]>' after instantiate-monomorphs}}
  // expected-note @below {{the witness of impl @Holds_i32 for requirement 0 of its trait rests on the allegation '!trait.claim<@Rule[tuple<i64, i64>]>', which nothing proves}}
  %m = trait.project %h[0] for [i1] : !trait.claim<@Holds[!T]> -> !trait.claim<@Rule[!trait.proj<@Holds[!T], "C", [i1]>]>
  %r = trait.func.call @use_rule(%m, %x) : (!trait.claim<@Rule[!trait.proj<@Holds[!T], "C", [i1]>]>, !trait.proj<@Holds[!T], "C", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Holds[i32], "C", [i1]>) -> i64 {
  %h = trait.allege @Holds[i32]
  %r = trait.func.call @f(%h, %x) : (!trait.claim<@Holds[i32]>, !trait.proj<@Holds[i32], "C", [i1]>) -> i64
  return %r : i64
}

// -----

// An impl reads its bound requirement off its premise's, which the premise's
// impl alleges: the refusal at the use names that allegation.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Mark[!S] {
  trait.method @value(!S) -> i64
}
trait.trait private @Base[!S] where [forall [!trait.bound<0>] -> @Mark[!trait.proj<@Base[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Base_i32 for @Base[i32]
    witnesses [#trait<witness requirement 0 by allege @Mark[i64]>] {
  trait.assoc_type @A<[!X]> = i64
}
trait.trait private @Outer[!S] where [forall [!trait.bound<0>] -> @Mark[!trait.proj<@Outer[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Outer_i32 for @Outer[i32] where [@Base[i32]]
    witnesses [#trait<witness requirement 0 by requirement 0 for [!trait.bound<0>] of where 0>] {
  trait.assoc_type @A<[!X]> = !trait.proj<@Base[i32], "A", [!X]>
}

func.func private @use(%m: !trait.claim<@Mark[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Mark[!M]::@value(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Outer[!T]>, %x: !trait.proj<@Outer[!T], "A", [i1]>) -> i64 {
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@Mark[i64]>' after instantiate-monomorphs}}
  // expected-note @below {{the witness of impl @Base_i32 for requirement 0 of its trait rests on the allegation '!trait.claim<@Mark[i64]>', which nothing proves}}
  %m = trait.project %h[0] for [i1] : !trait.claim<@Outer[!T]> -> !trait.claim<@Mark[!trait.proj<@Outer[!T], "A", [i1]>]>
  %r = trait.func.call @use(%m, %x) : (!trait.claim<@Mark[!trait.proj<@Outer[!T], "A", [i1]>]>, !trait.proj<@Outer[!T], "A", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Outer[i32], "A", [i1]>) -> i64 {
  %o = trait.allege @Outer[i32]
  %r = trait.func.call @f(%o, %x) : (!trait.claim<@Outer[i32]>, !trait.proj<@Outer[i32], "A", [i1]>) -> i64
  return %r : i64
}

// -----

// An impl reads its bound requirement off an application it alleges, which
// nothing proves: the refusal at the use names the allegation.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Sup0[!S] {
  trait.method @sup(!S) -> i64
}
trait.trait private @Sub0[!S] where [@Sup0[!S]] {}
trait.trait private @Has[!S] where [forall [!trait.bound<0>] -> @Sup0[!trait.proj<@Has[!S], "A", [!trait.bound<0>]>]] {
  trait.assoc_type @A<[!X]>
}
trait.impl private @Has_i32 for @Has[i32]
    witnesses [#trait<witness requirement 0 by requirement 0 of allege @Sub0[i64]>] {
  trait.assoc_type @A<[!X]> = i64
}

func.func private @use_sup(%m: !trait.claim<@Sup0[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Sup0[!M]::@sup(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Has[!T]>, %x: !trait.proj<@Has[!T], "A", [i1]>) -> i64 {
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@Sup0[i64]>' after instantiate-monomorphs}}
  // expected-note @below {{the witness of impl @Has_i32 for requirement 0 of its trait rests on the allegation '!trait.claim<@Sub0[i64]>', which nothing proves}}
  %m = trait.project %h[0] for [i1] : !trait.claim<@Has[!T]> -> !trait.claim<@Sup0[!trait.proj<@Has[!T], "A", [i1]>]>
  %r = trait.func.call @use_sup(%m, %x) : (!trait.claim<@Sup0[!trait.proj<@Has[!T], "A", [i1]>]>, !trait.proj<@Has[!T], "A", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Has[i32], "A", [i1]>) -> i64 {
  %h = trait.allege @Has[i32]
  %r = trait.func.call @f(%h, %x) : (!trait.claim<@Has[i32]>, !trait.proj<@Has[i32], "A", [i1]>) -> i64
  return %r : i64
}
