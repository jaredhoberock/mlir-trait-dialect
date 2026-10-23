// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// @I serves @A[i32] where an equality projecting through @A[i32] itself holds,
// and @I's own binding is what would make it hold. Selection reads the premise
// as the proof citing @I would, through selection: the projection meets
// @A[i32] again on the chain, so @I is no candidate and @A[i32]'s projection
// names no type, and a hypothesis equating it with a type is refused, as its
// allegation is (invalid_allegation_resting_on_an_impl_premise_through_its_own_application.mlir).
// A premise projection's associated-type argument is judged before any
// binding is substituted for the projection, so one naming no type, or
// resolving to evidence, refuses the premise whether or not the binding reads
// it.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @I(%s: !trait.claim<@A[i32]>, %e: !trait.claim<!trait.proj<@A[i32], "Item"> = i64>) { trait.assoc_type @Item = i64 }
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item">'}}
// expected-error @below {{equality '!trait.claim<!trait.proj<@A[i32], "Item"> = i64>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func @f(%h: !trait.claim<!trait.proj<@A[i32], "Item"> = i64>) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

// -----

!T = !trait.poly<0>
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Item }
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @I(%s: !trait.claim<@A[!T]>,
    %e: !trait.claim<!trait.proj<@A[!T], "Item", [!trait.proj<@B[!T], "Item">]> = i64>) {
  trait.assoc_type @Item<[!trait.poly<1>]> = i64
}
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item", [i32]>'}}
// expected-error @below {{equality '!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @C(%s: !trait.claim<@C[!T]>) {}
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Item }
trait.impl private @B_i32(%s: !trait.claim<@B[i32]>) { trait.assoc_type @Item = tuple<!trait.claim<@C[i32]>> }
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @I(%s: !trait.claim<@A[!T]>,
    %e: !trait.claim<!trait.proj<@A[!T], "Item", [!trait.proj<@B[!T], "Item">]> = i64>) {
  trait.assoc_type @Item<[!trait.poly<1>]> = i64
}
// expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item", [i32]>'}}
// expected-error @below {{equality '!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>' does not settle before erasure: a side resolves to no type a predicate states}}
func.func @f(%e: !trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>) { return }
