// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The inherited where-clause equality <@Wrap[i64]::Item = i64> is forwarded to a
// callee operand whose monomorphic instance retains the equality-claim
// parameter, so the surviving trait.assume has no other consumer. The equality
// holds only across two resolution hops: @Wrap_i64 binds Item to @Mid[i64]::Out,
// and @Mid_i64 binds that Out to i64. Settlement's fixed-point resolution
// grounds both hops; reducing the assume to a witness walks the same chain,
// minting one proj-resolve certificate per hop and composing them -- a single
// hop would leave @Mid[i64]::Out still spelled and the equality unproven.

// VERIFY: trait.assume !trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64

!S = !trait.poly<0>

trait.trait @Mid[!S] { trait.assoc_type @Out }
trait.impl @Mid_i64 for @Mid[i64] { trait.assoc_type @Out = i64 }

trait.trait @Wrap[!S] { trait.assoc_type @Item }
trait.impl @Wrap_i64 for @Wrap[i64] { trait.assoc_type @Item = !trait.proj<@Mid[i64], "Out"> }

trait.trait @Run[!S] {
  func.func private @go(!S) -> i64
}

// The callee's monomorphic instance keeps the equality-claim parameter on its
// ABI even though the body ignores it: the parameter carries the evidence
// across the call boundary.
func.func @need(%v: i64, %e: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64 {
  return %v : i64
}

// The closure-like impl: its where-clause carries the inherited equality; the
// method re-establishes it as an assume and forwards it as the call operand.
trait.impl @Run_gen for @Run[!S] where [@Wrap[!S], !trait.proj<@Wrap[!S], "Item"> = i64] {
  func.func @go(%x: !S) -> i64 {
    %e = trait.assume !trait.proj<@Wrap[!S], "Item"> = i64
    %v = arith.constant 7 : i64
    %r = trait.func.call @need(%v, %e)
      : (i64, !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64
    return %r : i64
  }
}

func.func @main() -> i64 {
  %c = trait.allege @Run[i64]
  %x = arith.constant 3 : i64
  %r = trait.method.call %c @Run[i64]::@go(%x) : (i64) -> i64
  return %r : i64
}

// The two-hop chain settles clean: the assume becomes the composed witness that
// proves <@Wrap[i64]::Item = i64>, the callee keeps the equality parameter, and
// no axiomatic assume survives.
// CHECK: func.func @need
// CHECK-NOT: trait.assume
// CHECK: func.func @main
