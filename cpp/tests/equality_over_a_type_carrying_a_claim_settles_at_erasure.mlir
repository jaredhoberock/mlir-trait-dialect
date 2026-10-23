// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An equality settles when both its sides resolve to one type that names no
// projection; a claim standing inside that type is a type like any other there.
// Each equality below holds -- reflexively, or because @Wrap_i32 binds
// @Wrap[i32]::Item to the claim type its other side spells -- and erasure's
// entry settles it, so each function lowers whichever pipeline hands erasure
// the module.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) {}
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {}
trait.trait private @Mark(%self: !trait.claim<@Mark[!trait.poly<0>]>) {}
trait.trait private @Wrap(%self: !trait.claim<@Wrap[!trait.poly<0>]>) { trait.assoc_type @Item }
trait.impl private @Wrap_i32(%self: !trait.claim<@Wrap[i32]>) { trait.assoc_type @Item = !trait.claim<@Mark[i32]> }

// A hypothesis whose two sides are one type carrying an unproven claim.
// CHECK-LABEL: func.func @reflexive_hypothesis() -> i64
func.func @reflexive_hypothesis(%c: !trait.claim<tuple<!trait.claim<@T[i32]>> = tuple<!trait.claim<@T[i32]>>>) -> i64 {
  %k = arith.constant 7 : i64
  return %k : i64
}

// A reflexive equality at an op result over a type carrying a claim beside a
// value.
// CHECK-LABEL: func.func @reflexive_result() -> i64
func.func @reflexive_result() -> i64 {
  %r = trait.witness refl : !trait.claim<tuple<i64, !trait.claim<@T[i32]>> = tuple<i64, !trait.claim<@T[i32]>>>
  %k = arith.constant 7 : i64
  return %k : i64
}

// A hypothesis equating a projection with the claim type it resolves to.
// CHECK-LABEL: func.func @projection_to_a_claim() -> i64
func.func @projection_to_a_claim(%e: !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>) -> (i64, !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>) {
  %c = arith.constant 7 : i64
  return %c, %e : i64, !trait.claim<!trait.proj<@Wrap[i32], "Item"> = !trait.claim<@Mark[i32]>>
}
