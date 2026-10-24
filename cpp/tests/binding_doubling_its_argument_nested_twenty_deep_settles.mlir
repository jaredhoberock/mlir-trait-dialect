// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: %python %S/Inputs/expand_repeats.py %s | timeout 120 mlir-opt -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @I and @I2 each bind their associated type to a pair of its argument, and
// !P20 and !Q20 nest a projection of each twenty deep in its own argument. Both
// read to one normal form, a tree of 2^20 leaves spelled by twenty-one
// distinct types: the width a reading is bounded by counts distinct types, so
// the hypothesis that the two are one type settles, and erasure removes it.

// CHECK: func.func @f() {

!T = !trait.poly<0>
trait.trait private @A(%s: !trait.claim<@A[!T]>) { trait.assoc_type @Item<[!trait.poly<1>]> }
trait.impl private @I(%s: !trait.claim<@A[i32]>) { trait.assoc_type @Item<[!trait.poly<0>]> = tuple<!trait.poly<0>, !trait.poly<0>> }
trait.trait private @A2(%s: !trait.claim<@A2[!T]>) { trait.assoc_type @Item2<[!trait.poly<1>]> }
trait.impl private @I2(%s: !trait.claim<@A2[i32]>) { trait.assoc_type @Item2<[!trait.poly<0>]> = tuple<!trait.poly<0>, !trait.poly<0>> }
!P0 = i64
!Q0 = i64
// REPEAT 0 19: !P{k+1} = !trait.proj<@A[i32], "Item", [!P{k}]>
// REPEAT 0 19: !Q{k+1} = !trait.proj<@A2[i32], "Item2", [!Q{k}]>
func.func @f(%e: !trait.claim<!P20 = !Q20>) { return }
