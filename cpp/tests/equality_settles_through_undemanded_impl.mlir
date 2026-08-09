// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A composed equality whose two projection endpoints resolve to one ground
// spelling through a common middle strand -- and every impl in that chain is
// one nothing else in the module demands. @Has_w1 and @Has_w2 carry each
// wrapper's @Out to @Has[i32]::Out, and @Has_mid carries that middle to i64;
// composing the two proj-resolve premises names the equality their congruence
// closure entails, @Has[tuple<i32>]::Out = @Has[tuple<i32, i32>]::Out. The
// equality crosses the call to the generic callee and survives to the leftover
// check, where the rounds recorded no outcome for any of the three impls,
// because an equality's endpoints are opaque to them. Settlement puts the
// endpoint projections to impl selection itself: each unique unconditional impl
// is selected, the chain resolves to i64 on both sides, the endpoints meet, and
// the whole module lowers clean.

// VERIFY: trait.witness compose

!U = !trait.poly<1>

trait.trait @Has[!trait.poly<0>] { trait.assoc_type @Out }

trait.impl @Has_w1  for @Has[tuple<!U>]     { trait.assoc_type @Out = !trait.proj<@Has[!U], "Out"> }
trait.impl @Has_w2  for @Has[tuple<!U, !U>] { trait.assoc_type @Out = !trait.proj<@Has[!U], "Out"> }
trait.impl @Has_mid for @Has[i32]           { trait.assoc_type @Out = i64 }

func.func @gen(%c: !trait.claim<!trait.proj<@Has[tuple<!U>], "Out"> = !trait.proj<@Has[tuple<!U, !U>], "Out">>) -> () {
  return
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: trait.witness
// CHECK-NOT: trait.claim
// CHECK-NOT: trait.proj
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
