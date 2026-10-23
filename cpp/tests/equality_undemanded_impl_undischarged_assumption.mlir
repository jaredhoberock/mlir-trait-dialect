// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' 2>&1 | FileCheck %s --check-prefix=ERASE

// The refusal half of the undemanded settlement. The structure matches
// equality_settles_through_undemanded_impl.mlir, but the middle impl @Has_mid
// is conditional on @X[i32], for which the module carries no impl. Settlement
// puts the endpoint projections to impl selection; selecting @Has_mid to carry
// @Has[i32]::Out to i64 demands its assumption @X[i32], which is unsatisfiable,
// so selection refuses it and @Has[i32]::Out stays spelled. Instantiation
// refuses that projection where the witnesses spell it; erasure, run alone,
// refuses the coerced equality, whose endpoints do not meet at a ground type --
// an equality settles only through impls whose where-bounds hold, never through
// @Has_mid on the strength of its binding alone. Supplying an @X[i32] impl is
// what discharges the assumption and lets the equality settle.

// VERIFY: trait.coerce

!U = !trait.poly<1>

trait.trait private @X(%self: !trait.claim<@X[!trait.poly<0>]>) {}
trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }

trait.impl private @Has_w1(%self: !trait.claim<@Has[tuple<!trait.poly<0>>]>) { trait.assoc_type @Out = !trait.proj<@Has[!trait.poly<0>], "Out"> }
trait.impl private @Has_w2(%self: !trait.claim<@Has[tuple<!trait.poly<0>, !trait.poly<0>>]>) { trait.assoc_type @Out = !trait.proj<@Has[!trait.poly<0>], "Out"> }
trait.impl private @Has_mid(%self: !trait.claim<@Has[i32]>, %x: !trait.claim<@X[i32]>) { trait.assoc_type @Out = i64 }

func.func private @gen(%c: !trait.claim<!trait.proj<@Has[tuple<!trait.poly<0>>], "Out"> = !trait.proj<@Has[tuple<!trait.poly<0>, !trait.poly<0>>], "Out">>) -> () {
  return
}

// CHECK: error: no impl with satisfiable assumptions for '!trait.proj<@Has[i32], "Out">'
// CHECK: error: unresolved projection '!trait.proj<@Has[i32], "Out">' after instantiate-monomorphs
// ERASE: error: equality '!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32, i32>], "Out">>' does not settle before erasure: both its sides resolve to '!trait.proj<@Has[i32], "Out">', which is not a ground type
func.func @main() -> () {
  %p1 = trait.witness proj_resolve !trait.proj<@Has[tuple<i32>], "Out"> resolves !trait.proj<@Has[i32], "Out"> by @Has_w1[i32]
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[i32], "Out">>
  %p2 = trait.witness proj_resolve !trait.proj<@Has[tuple<i32, i32>], "Out"> resolves !trait.proj<@Has[i32], "Out"> by @Has_w2[i32]
    : !trait.claim<!trait.proj<@Has[tuple<i32, i32>], "Out"> = !trait.proj<@Has[i32], "Out">>
  %r = trait.witness refl
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32>], "Out">>
  %c = trait.coerce %r
    : !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32>], "Out">>
    to !trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32, i32>], "Out">>
    via (%p1, %p2) : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[i32], "Out">>, !trait.claim<!trait.proj<@Has[tuple<i32, i32>], "Out"> = !trait.proj<@Has[i32], "Out">>)
  trait.func.call @gen(%c) : (!trait.claim<!trait.proj<@Has[tuple<i32>], "Out"> = !trait.proj<@Has[tuple<i32, i32>], "Out">>) -> ()
  return
}
