// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' -verify-diagnostics

// Selection settles @A[i32], whose binding of Item branches, and refuses the
// allegation of @B whose argument projects Item at an argument no impl
// serves. Naming that refusal reads the application through what selection
// settled, and that reading, like selection's, reads a projection's arguments
// before it projects: the argument never settles, so the branching binding is
// never expanded, and the refusal named is the argument's.

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!T>]>, !trait.proj<@A[i32], "Item", [tuple<!T, !T>]>>} {
}
trait.trait private @B(!trait.claim<@B[!T]>) assoc {Out = 0}
func.func @f() {
  %s = trait.allege @A[i32]
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@B[i64], "Out">'}}
  // expected-error @below {{unproven monomorphic claim '!trait.claim<@B[!trait.proj<@A[i32], "Item", [!trait.proj<@B[i64], "Out">]>]>' after instantiate-monomorphs}}
  %p = trait.allege @B[!trait.proj<@A[i32], "Item", [!trait.proj<@B[i64], "Out">]>]
  return
}

// -----

// The same refusal with @A[i32] settled through another of its associated
// types, the branching projection beside it in a tuple.

!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Seed = 0, Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Seed = i64, Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>} {
}
trait.trait private @X(!trait.claim<@X[!T]>) assoc {Out = 0}
trait.trait private @B(!trait.claim<@B[!T]>)
func.func @f() {
  // expected-error @below {{no impl with satisfiable assumptions for '!trait.proj<@X[i32], "Out">'}}
  // expected-error @below {{unproven monomorphic claim}}
  %e = trait.allege @B[tuple<!trait.proj<@A[i32], "Seed">, !trait.proj<@A[i32], "Item", [!trait.proj<@X[i32], "Out">]>>]
  return
}
