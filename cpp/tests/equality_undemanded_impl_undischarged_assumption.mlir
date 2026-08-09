// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The refusal half of the undemanded settlement. The structure matches
// equality_settles_through_undemanded_impl.mlir, but the middle impl @Has_mid
// is conditional on @X[i32], for which the module carries no impl. Settlement
// puts the endpoint projections to impl selection; selecting @Has_mid to carry
// @Has[i32]::Out to i64 demands its assumption @X[i32], which is unsatisfiable,
// so selection refuses it and @Has[i32]::Out stays spelled. The endpoints do not
// meet at a ground type, and the composed equality is reported as a leftover --
// the settlement resolves only through impls whose where-bounds hold, never
// through @Has_mid on the strength of its binding alone. Supplying an @X[i32]
// impl is what discharges the assumption and lets the equality settle.

// VERIFY: trait.witness compose

!U = !trait.poly<1>

trait.trait @X[!trait.poly<9>] {}
trait.trait @Has[!trait.poly<0>] { trait.assoc_type @Out }

trait.impl @Has_w1  for @Has[tuple<!U>]     { trait.assoc_type @Out = !trait.proj<@Has[!U], "Out"> }
trait.impl @Has_w2  for @Has[tuple<!U, !U>] { trait.assoc_type @Out = !trait.proj<@Has[!U], "Out"> }
trait.impl @Has_mid for @Has[i32] where [@X[i32]] { trait.assoc_type @Out = i64 }

func.func @gen(%c: !trait.claim<!trait.proj<@Has[tuple<!U>], "Out"> = !trait.proj<@Has[tuple<!U, !U>], "Out">>) -> () {
  return
}

// CHECK: unproven monomorphic claim '!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32, i32>], "Out">>' after instantiate-monomorphs
func.func @main() -> () {
  %p1 = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves !trait.proj<@Has[i32], "Out"> by @Has_w1
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[i32], "Out">>
  %p2 = trait.witness proj_resolve !trait.proj<@Has[tuple<i32, i32>], "Out"> resolves !trait.proj<@Has[i32], "Out"> by @Has_w2
    : !trait.claim<!trait.proj<@Has[tuple<i32, i32>], "Out"> = !trait.proj<@Has[i32], "Out">>
  %c = trait.witness compose(%p1, %p2)
    : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[i32], "Out">>, !trait.claim<!trait.proj<@Has[tuple<i32, i32>], "Out"> = !trait.proj<@Has[i32], "Out">>)
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32, i32>], "Out">>
  trait.func.call @gen(%c) : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32, i32>], "Out">>) -> ()
  return
}
