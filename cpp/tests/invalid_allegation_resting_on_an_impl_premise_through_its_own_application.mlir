// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// The allegations of the equalities
// invalid_hypothesis_resting_on_an_impl_premise_through_its_own_application.mlir
// assumes, with one verdict: @I's premise projects through @A[i32], which
// selection meets again on the chain, so @I is no candidate and the allegation
// is refused naming that, whether the premise's own argument is a type no impl
// serves or one an impl serves.

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item }
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @I(%s: !trait.claim<@A[i32]>, %e: !trait.claim<!trait.proj<@A[i32], "Item"> = i64>) { trait.assoc_type @Item = i64 }
func.func private @need(!trait.claim<!trait.proj<@A[i32], "Item"> = i64>)
func.func @f() {
  // expected-error @below {{'trait.allege' op no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item">'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@A[i32], "Item"> = i64>' after instantiate-monomorphs}}
  %h = trait.allege !trait.proj<@A[i32], "Item"> = i64
  func.call @need(%h) : (!trait.claim<!trait.proj<@A[i32], "Item"> = i64>) -> ()
  return
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
func.func private @need(!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>)
func.func @f() {
  // expected-error @below {{'trait.allege' op no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item", [i32]>'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@A[i32], "Item", [i32]> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>) -> ()
  return
}

// -----

!T = !trait.poly<0>
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Item }
trait.impl private @B_i32(%s: !trait.claim<@B[i32]>) { trait.assoc_type @Item = i32 }
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
// expected-note @below {{unsatisfiable candidate}}
trait.impl private @I(%s: !trait.claim<@A[!T]>,
    %e: !trait.claim<!trait.proj<@A[!T], "Item", [!trait.proj<@B[!T], "Item">]> = i64>) {
  trait.assoc_type @Item<[!trait.poly<1>]> = i64
}
func.func private @need(!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>)
func.func @f() {
  // expected-error @below {{'trait.allege' op no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item", [i32]>'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>' after instantiate-monomorphs}}
  %e = trait.allege !trait.proj<@A[i32], "Item", [i32]> = i64
  func.call @need(%e) : (!trait.claim<!trait.proj<@A[i32], "Item", [i32]> = i64>) -> ()
  return
}
