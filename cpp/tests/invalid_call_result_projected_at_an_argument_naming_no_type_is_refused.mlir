// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -split-input-file -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' 2>&1 | FileCheck %s

// @f returns @A[i32]'s Item at @B[T]'s Out, and @I binds Item to i64 whatever
// its argument. At the call, T is i64: @B[i64]'s Out has no impl, and in the
// second module it resolves to a claim. Either way the argument names no type,
// so the projection names none, and the call's substitution takes no step
// at it: the call stays standing, named by the projection no impl serves,
// rather than lowering to i64. In the third module @B[i64]'s Out is i32, and
// the call lowers.

// CHECK: error: unresolved projection '!trait.proj<@A[i32], "Item", [!trait.proj<@B[i64], "Out">]>' after instantiate-monomorphs
// CHECK: error: no impl with satisfiable assumptions for '!trait.proj<@B[i64], "Out">'
!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item<[!trait.poly<1>]> }
trait.impl private @I(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item<[!T]> = i64 }
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Out }
!P = !trait.proj<@A[i32], "Item", [!trait.proj<@B[!T], "Out">]>
!Q = !trait.proj<@A[i32], "Item", [!trait.proj<@B[i64], "Out">]>
func.func private @f(%x: !T) -> !P { %v = ub.poison : !P return %v : !P }
func.func @run() { %c = arith.constant 7 : i64 %r = trait.func.call @f(%c) : (i64) -> !Q return }

// -----

// CHECK: error: unproven monomorphic claim '!trait.claim<@Mark[i64]>' after instantiate-monomorphs
// CHECK: error: unresolved projection '!trait.proj<@A[i32], "Item", [!trait.claim<@Mark[i64]>]>' after instantiate-monomorphs
!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item<[!trait.poly<1>]> }
trait.impl private @I(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item<[!T]> = i64 }
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Out }
trait.trait private @Mark(%s: !trait.claim<@Mark[!T]>) {}
trait.impl private @J(%s: !trait.claim<@B[i64]>) { trait.assoc_type @Out = !trait.claim<@Mark[i64]> }
!P = !trait.proj<@A[i32], "Item", [!trait.proj<@B[!T], "Out">]>
!Q = !trait.proj<@A[i32], "Item", [!trait.proj<@B[i64], "Out">]>
func.func private @f(%x: !T) -> !P { %v = ub.poison : !P return %v : !P }
func.func @run() { %c = arith.constant 7 : i64 %r = trait.func.call @f(%c) : (i64) -> !Q return }

// -----

// CHECK-NOT: error
!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item<[!trait.poly<1>]> }
trait.impl private @I(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item<[!T]> = i64 }
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Out }
trait.impl private @J(%s: !trait.claim<@B[i64]>) { trait.assoc_type @Out = i32 }
!P = !trait.proj<@A[i32], "Item", [!trait.proj<@B[!T], "Out">]>
!Q = !trait.proj<@A[i32], "Item", [!trait.proj<@B[i64], "Out">]>
func.func private @f(%x: !T) -> !P { %v = ub.poison : !P return %v : !P }
func.func @run() { %c = arith.constant 7 : i64 %r = trait.func.call @f(%c) : (i64) -> !Q return }
