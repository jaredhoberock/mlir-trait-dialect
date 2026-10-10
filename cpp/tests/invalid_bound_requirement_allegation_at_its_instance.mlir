// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// An impl's evidence method for a quantified requirement alleges its
// conclusion, and nothing proves it at the instance the use reaches. The use is
// replaced by the method's body, and the allegation it inlines is refused where
// it is written, naming the claim: selection names that it has no impl when it
// is first asked, and the stage's exit walk that it stands unproven.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Rule(%self: !trait.claim<@Rule[!S]>) {
  trait.method @size(!S) -> i64
}
trait.trait private @Holds(%self: !trait.claim<@Holds[!S]>) {
  trait.assoc_type @C<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[!S], "C", [!X]>]>
}
trait.impl private @Holds_i32(%self: !trait.claim<@Holds[i32]>) {
  trait.assoc_type @C<[!X]> = tuple<i64, i64>
  trait.method @requirement_0() -> !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!X]>]> {
    // expected-error @below {{no impl with satisfiable assumptions for '!trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [i1]>]>'}}
    // expected-error @below {{unproven monomorphic claim '!trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [i1]>]>' after instantiate-monomorphs}}
    %r = trait.allege @Rule[!trait.proj<@Holds[i32], "C", [!X]>]
    trait.return %r : !trait.claim<@Rule[!trait.proj<@Holds[i32], "C", [!X]>]>
  }
}

func.func private @use_rule(%m: !trait.claim<@Rule[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Rule[!M]::@size(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Holds[!T]>, %x: !trait.proj<@Holds[!T], "C", [i1]>) -> i64 {
  %m = trait.method.call %h @Holds[!T]::@requirement_0() : () -> !trait.claim<@Rule[!trait.proj<@Holds[!T], "C", [i1]>]>
  %r = trait.func.call @use_rule(%m, %x) : (!trait.claim<@Rule[!trait.proj<@Holds[!T], "C", [i1]>]>, !trait.proj<@Holds[!T], "C", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Holds[i32], "C", [i1]>) -> i64 {
  %h = trait.allege @Holds[i32]
  %r = trait.func.call @f(%h, %x) : (!trait.claim<@Holds[i32]>, !trait.proj<@Holds[i32], "C", [i1]>) -> i64
  return %r : i64
}

// -----

// An impl's evidence method calls its where argument's, whose impl alleges the
// conclusion: both calls are replaced by their methods' bodies, and the
// allegation the inner one inlines is refused where it is written, and the
// coercion respelling it stands unproven with it.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Mark(%self: !trait.claim<@Mark[!S]>) {
  trait.method @value(!S) -> i64
}
trait.trait private @Base(%self: !trait.claim<@Base[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Base[!S], "A", [!X]>]>
}
trait.impl private @Base_i32(%self: !trait.claim<@Base[i32]>) {
  trait.assoc_type @A<[!X]> = i64
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Base[i32], "A", [!X]>]> {
    // expected-error @below {{no impl with satisfiable assumptions for '!trait.claim<@Mark[!trait.proj<@Base[i32], "A", [i1]>]>'}}
    // expected-error @below {{unproven monomorphic claim '!trait.claim<@Mark[!trait.proj<@Base[i32], "A", [i1]>]>' after instantiate-monomorphs}}
    %r = trait.allege @Mark[!trait.proj<@Base[i32], "A", [!X]>]
    trait.return %r : !trait.claim<@Mark[!trait.proj<@Base[i32], "A", [!X]>]>
  }
}
trait.trait private @Outer(%self: !trait.claim<@Outer[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Outer[!S], "A", [!X]>]>
}
trait.impl private @Outer_i32(%self: !trait.claim<@Outer[i32]>, %base: !trait.claim<@Base[i32]>) {
  trait.assoc_type @A<[!X]> = !trait.proj<@Base[i32], "A", [!X]>
  trait.method @requirement_0() -> !trait.claim<@Mark[!trait.proj<@Outer[i32], "A", [!X]>]> {
    %r = trait.method.call %base @Base[i32]::@requirement_0() : () -> !trait.claim<@Mark[!trait.proj<@Base[i32], "A", [!X]>]>
    %e = trait.witness proj_resolve !trait.proj<@Outer[i32], "A", [!X]> resolves !trait.proj<@Base[i32], "A", [!X]> by @Outer_i32 given(%base)
      : (!trait.claim<@Base[i32]>)
      : !trait.claim<!trait.proj<@Outer[i32], "A", [!X]> = !trait.proj<@Base[i32], "A", [!X]>>
    // expected-error @below {{unproven monomorphic claim '!trait.claim<@Mark[!trait.proj<@Outer[i32], "A", [i1]>]>' after instantiate-monomorphs}}
    %c = trait.coerce %r : !trait.claim<@Mark[!trait.proj<@Base[i32], "A", [!X]>]> to !trait.claim<@Mark[!trait.proj<@Outer[i32], "A", [!X]>]> via (%e)
      : (!trait.claim<!trait.proj<@Outer[i32], "A", [!X]> = !trait.proj<@Base[i32], "A", [!X]>>)
    trait.return %c : !trait.claim<@Mark[!trait.proj<@Outer[i32], "A", [!X]>]>
  }
}

func.func private @use(%m: !trait.claim<@Mark[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Mark[!M]::@value(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Outer[!T]>, %x: !trait.proj<@Outer[!T], "A", [i1]>) -> i64 {
  %m = trait.method.call %h @Outer[!T]::@requirement_0() : () -> !trait.claim<@Mark[!trait.proj<@Outer[!T], "A", [i1]>]>
  %r = trait.func.call @use(%m, %x) : (!trait.claim<@Mark[!trait.proj<@Outer[!T], "A", [i1]>]>, !trait.proj<@Outer[!T], "A", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Outer[i32], "A", [i1]>) -> i64 {
  %o = trait.allege @Outer[i32]
  %r = trait.func.call @f(%o, %x) : (!trait.claim<@Outer[i32]>, !trait.proj<@Outer[i32], "A", [i1]>) -> i64
  return %r : i64
}

// -----

// An impl's evidence method projects its conclusion off an application it
// alleges, which nothing proves: the allegation and the projection waiting on
// it are refused where they are written.

!S = !trait.poly<0>
!X = !trait.poly<1>
!T = !trait.poly<2>
!M = !trait.poly<3>

trait.trait private @Sup0(%self: !trait.claim<@Sup0[!S]>) {
  trait.method @sup(!S) -> i64
}
trait.trait private @Sub0(%self: !trait.claim<@Sub0[!S]>) -> !trait.claim<@Sup0[!S]> {}
trait.trait private @Has(%self: !trait.claim<@Has[!S]>) {
  trait.assoc_type @A<[!X]>
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[!S], "A", [!X]>]>
}
trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) {
  trait.assoc_type @A<[!X]> = i64
  trait.method @requirement_0() -> !trait.claim<@Sup0[!trait.proj<@Has[i32], "A", [!X]>]> {
    // expected-error @below {{no impl with satisfiable assumptions for '!trait.claim<@Sub0[!trait.proj<@Has[i32], "A", [i1]>]>'}}
    // expected-error @below {{unproven monomorphic claim '!trait.claim<@Sub0[!trait.proj<@Has[i32], "A", [i1]>]>' after instantiate-monomorphs}}
    %s = trait.allege @Sub0[!trait.proj<@Has[i32], "A", [!X]>]
    // expected-error @below {{unproven monomorphic claim '!trait.claim<@Sup0[!trait.proj<@Has[i32], "A", [i1]>]>' after instantiate-monomorphs}}
    %r = trait.project %s[0] : !trait.claim<@Sub0[!trait.proj<@Has[i32], "A", [!X]>]> -> !trait.claim<@Sup0[!trait.proj<@Has[i32], "A", [!X]>]>
    trait.return %r : !trait.claim<@Sup0[!trait.proj<@Has[i32], "A", [!X]>]>
  }
}

func.func private @use_sup(%m: !trait.claim<@Sup0[!M]>, %x: !M) -> i64 {
  %r = trait.method.call %m @Sup0[!M]::@sup(%x) : (!M) -> i64
  return %r : i64
}

func.func private @f(%h: !trait.claim<@Has[!T]>, %x: !trait.proj<@Has[!T], "A", [i1]>) -> i64 {
  %m = trait.method.call %h @Has[!T]::@requirement_0() : () -> !trait.claim<@Sup0[!trait.proj<@Has[!T], "A", [i1]>]>
  %r = trait.func.call @use_sup(%m, %x) : (!trait.claim<@Sup0[!trait.proj<@Has[!T], "A", [i1]>]>, !trait.proj<@Has[!T], "A", [i1]>) -> i64
  return %r : i64
}

func.func @main(%x: !trait.proj<@Has[i32], "A", [i1]>) -> i64 {
  %h = trait.allege @Has[i32]
  %r = trait.func.call @f(%h, %x) : (!trait.claim<@Has[i32]>, !trait.proj<@Has[i32], "A", [i1]>) -> i64
  return %r : i64
}
