// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=SHAPE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An equality endpoint may be a composite that carries a projection nested
// inside it -- the resolved side of `type Out = tuple<Self::Item>`, where the
// projection sits inside the tuple rather than at its top. @Inner_i64 binds Item
// to i64, so <tuple<@Inner[i64]::Item> = tuple<i64>> holds through the one hop
// nested in the tuple. Settlement resolves ground projections anywhere in an
// endpoint, descending composites, so it grounds the endpoint at tuple<i64>;
// reducing the surviving equality assume to a witness descends the same
// composite and mints the proj-resolve leaf for the nested @Inner[i64]::Item
// hop, then composes it to the endpoint equality. A mint that inspected only the
// endpoint's top level would find no projection there, mint nothing, and refuse
// the assume it had already judged settled.

// The composition witness the reduction mints has this shape: the nested
// proj-resolve leaf grounding @Inner[i64]::Item, and a compose whose ground
// congruence closure lifts that equality through the tuple to the endpoint
// equality.
// SHAPE: trait.witness compose(%{{[0-9]+}}) : (!trait.claim<!trait.proj<@Inner[i64], "Item"> = i64>) : !trait.claim<tuple<!trait.proj<@Inner[i64], "Item">> = tuple<i64>>

!S = !trait.poly<0>

trait.trait @Inner[!S] { trait.assoc_type @Item }
trait.impl @Inner_i64 for @Inner[i64] { trait.assoc_type @Item = i64 }

// The composition witness the reduction mints, written out by hand so its shape
// and the verifier's acceptance of a projection nested in a composite endpoint
// are both pinned. The coerce keeps the witness live so parse-time verification
// judges it.
func.func @evidence(%p: tuple<!trait.proj<@Inner[i64], "Item">>) -> tuple<i64> {
  %w = trait.witness proj_resolve !trait.proj<@Inner[i64], "Item"> resolves i64 by @Inner_i64
    : !trait.claim<!trait.proj<@Inner[i64], "Item"> = i64>
  %c = trait.witness compose(%w)
    : (!trait.claim<!trait.proj<@Inner[i64], "Item"> = i64>)
    : !trait.claim<tuple<!trait.proj<@Inner[i64], "Item">> = tuple<i64>>
  %v = trait.coerce %p : tuple<!trait.proj<@Inner[i64], "Item">> to tuple<i64> via (%c)
    : (!trait.claim<tuple<!trait.proj<@Inner[i64], "Item">> = tuple<i64>>)
  return %v : tuple<i64>
}

trait.trait @Run[!S] {
  func.func private @go(!S) -> i64
}

// The callee's monomorphic instance keeps the equality-claim parameter on its
// ABI even though the body ignores it: the parameter carries the evidence across
// the call boundary, so the assume that supplies it has no other consumer and
// survives to the leftover check as a monomorphic equality whose endpoint is a
// composite carrying a projection.
func.func @need(%v: i64, %e: !trait.claim<tuple<!trait.proj<@Inner[!S], "Item">> = tuple<i64>>) -> i64 {
  return %v : i64
}

// The closure-like impl: its where-clause carries the inherited equality; the
// method re-establishes it as an assume and forwards it as the call operand.
trait.impl @Run_gen for @Run[!S] where [@Inner[!S], tuple<!trait.proj<@Inner[!S], "Item">> = tuple<i64>] {
  func.func @go(%x: !S) -> i64 {
    %e = trait.assume tuple<!trait.proj<@Inner[!S], "Item">> = tuple<i64>
    %v = arith.constant 7 : i64
    %r = trait.func.call @need(%v, %e)
      : (i64, !trait.claim<tuple<!trait.proj<@Inner[!S], "Item">> = tuple<i64>>) -> i64
    return %r : i64
  }
}

func.func @main() -> i64 {
  %c = trait.allege @Run[i64]
  %x = arith.constant 3 : i64
  %r = trait.method.call %c @Run[i64]::@go(%x) : (i64) -> i64
  return %r : i64
}

// The composite endpoint settles clean: the assume reduces to the composition
// witness above, the callee keeps the equality parameter, and no axiomatic
// assume survives.
// CHECK: func.func @main
// CHECK-NOT: trait.assume
