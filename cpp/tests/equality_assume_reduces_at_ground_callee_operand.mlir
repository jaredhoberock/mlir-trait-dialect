// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An impl inherits a where-clause equality and re-establishes it inside its
// method as an equality-arm trait.assume, then forwards that assume to a callee
// whose monomorphic instance retains the equality-claim parameter. When the
// assume also fed a trait.coerce the coerce folds at ground and the now-dead
// assume is eliminated; here the assume feeds a live call operand instead, so
// nothing consumes it. The projection @Wrap[i64]::Item ground-resolves to i64
// through the unconditional impl, so the equality <@Wrap[i64]::Item = i64> is
// definitionally true: settlement replaces the surviving assume with the
// proj-resolve witness that proves it -- the same evidence a caller mints where
// the equality is first established -- and no trait.assume reaches legalization.

// VERIFY: trait.assume !trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64

!S = !trait.poly<0>

trait.trait @Wrap[!S] { trait.assoc_type @Item }

trait.impl @Wrap_i64 for @Wrap[i64] { trait.assoc_type @Item = i64 }

trait.trait @Run[!S] {
  func.func private @go(!S) -> i64
}

// The callee's monomorphic instance keeps the equality-claim parameter on its
// ABI even though the body ignores it: the parameter design carries the
// evidence across the call boundary.
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

// The concrete instance settles clean: the assume becomes the witness that
// proves <@Wrap[i64]::Item = i64>, the callee keeps the equality parameter, and
// no axiomatic assume survives.
// CHECK: func.func @need
// CHECK-NOT: trait.assume
// CHECK: func.func @main
