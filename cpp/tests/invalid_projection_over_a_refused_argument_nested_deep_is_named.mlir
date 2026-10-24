// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | not timeout 120 mlir-opt -split-input-file -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/expand_repeats.py %s | not timeout 120 mlir-opt -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// No impl serves @A[i32], and !P64 nests a projection over it sixty-four deep
// in its own argument; no impl serves @B at all, and !Q25 nests one twenty-five
// deep in its own head's argument. A projection is offered once its arguments
// are read, and one whose argument names no type is never offered, so each
// level is read once and stands: the innermost refusal is named, and the
// hypothesis is refused by its side naming no type.

// CHECK: no impl with satisfiable assumptions for '!trait.proj<@A[i32], "Item", [i64]>'
// CHECK: does not settle before erasure: a side resolves to no type a predicate states
// CHECK: no impl with satisfiable assumptions for '!trait.proj<@B[i64], "Item">'
// CHECK: does not settle before erasure: a side resolves to no type a predicate states

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) {
  trait.assoc_type @Item<[!trait.poly<1>]>
}
trait.impl private @I(%s: !trait.claim<@A[i64]>) {
  trait.assoc_type @Item<[!trait.poly<0>]> = !trait.poly<0>
}
!P0 = i64
// REPEAT 0 63: !P{k+1} = !trait.proj<@A[i32], "Item", [!P{k}]>
func.func @f(%e: !trait.claim<!P64 = i64>) { return }

// -----

!T = !trait.poly<0>
trait.trait private @B(%s: !trait.claim<@B[!T]>) { trait.assoc_type @Item }
!Q0 = i64
// REPEAT 0 24: !Q{k+1} = !trait.proj<@B[!Q{k}], "Item">
func.func @g(%e: !trait.claim<!Q25 = i64>) { return }
