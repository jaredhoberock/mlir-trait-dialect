// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: not timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// @f's first allegation has selection settle @I for @A[i32]; its second demands
// @B at a projection of @A[i32]'s Item whose argument, @Wrap[i32]'s Out,
// resolves to a type carrying a claim, which names no type. No step is taken
// at such an argument, whichever reading asks, so selection refuses the
// demand without expanding @I's binding, and its naming of that refusal reads
// what selection settled without expanding it either: the refusal is named by
// the argument, promptly, whether the binding doubles its argument, branches,
// or ignores it.

// CHECK: error: 'trait.allege' op a predicate states types, and '!trait.claim<@B[!trait.proj<@A[i32], "Item", [!trait.proj<@Wrap[i32], "Out">]>]>' resolves to '!trait.proj<@A[i32], "Item", [tuple<!trait.claim<@Mark[i32]>>]>', which carries a claim
!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>)
trait.trait private @Wrap(!trait.claim<@Wrap[!T]>) assoc {Out = 0}
trait.impl private @Wrap_i32(%s: !trait.claim<@Wrap[i32]>) bindings {Out = tuple<!trait.claim<@Mark[i32]>>} {
}
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>} {
}
trait.trait private @B(!trait.claim<@B[!T]>)
func.func @f() {
  %s = trait.allege @A[i32]
  %p = trait.allege @B[!trait.proj<@A[i32], "Item", [!trait.proj<@Wrap[i32], "Out">]>]
  return
}

// -----

// CHECK: error: 'trait.allege' op a predicate states types, and '!trait.claim<@B[!trait.proj<@A[i32], "Item", [!trait.proj<@Wrap[i32], "Out">]>]>' resolves to '!trait.proj<@A[i32], "Item", [tuple<!trait.claim<@Mark[i32]>>]>', which carries a claim
!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>)
trait.trait private @Wrap(!trait.claim<@Wrap[!T]>) assoc {Out = 0}
trait.impl private @Wrap_i32(%s: !trait.claim<@Wrap[i32]>) bindings {Out = tuple<!trait.claim<@Mark[i32]>>} {
}
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>, !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>, !trait.poly<0>>]>>} {
}
trait.trait private @B(!trait.claim<@B[!T]>)
func.func @f() {
  %s = trait.allege @A[i32]
  %p = trait.allege @B[!trait.proj<@A[i32], "Item", [!trait.proj<@Wrap[i32], "Out">]>]
  return
}

// -----

// CHECK: error: 'trait.allege' op a predicate states types, and '!trait.claim<@B[!trait.proj<@A[i32], "Item", [!trait.proj<@Wrap[i32], "Out">]>]>' resolves to '!trait.proj<@A[i32], "Item", [tuple<!trait.claim<@Mark[i32]>>]>', which carries a claim
!T = !trait.poly<0>
trait.trait private @Mark(!trait.claim<@Mark[!T]>)
trait.trait private @Wrap(!trait.claim<@Wrap[!T]>) assoc {Out = 0}
trait.impl private @Wrap_i32(%s: !trait.claim<@Wrap[i32]>) bindings {Out = tuple<!trait.claim<@Mark[i32]>>} {
}
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = i64} {
}
trait.trait private @B(!trait.claim<@B[!T]>)
trait.impl private @IB(%s: !trait.claim<@B[i64]>) {
}
func.func @f() {
  %s = trait.allege @A[i32]
  %p = trait.allege @B[!trait.proj<@A[i32], "Item", [!trait.proj<@Wrap[i32], "Out">]>]
  return
}
