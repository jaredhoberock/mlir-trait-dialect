// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The refusal guard on the two-hop reduction. The chain's first hop grounds
// (@Wrap_i64 binds Item to @Mid[i64]::Out), but the second hop's impl @Mid_i64
// is conditional on @X[i64], for which the module carries no impl. A projection
// resolves only through an impl whose obligations hold, so @Mid[i64]::Out stays
// spelled and the premise's endpoints never meet at a ground type. Discharging
// the premise is what selection does, so @Run_gen is no candidate for the
// demand and the allegation has no impl -- the equality is never reduced to a
// witness on the strength of @Mid_i64's binding alone. Supplying an @X[i64]
// impl is what discharges the assumption and lets the premise hold.

// VERIFY: trait.assume !trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64

!S = !trait.poly<0>

trait.trait private @X[!trait.poly<9>] {}
trait.trait private @Mid[!S] { trait.assoc_type @Out }
trait.impl private @Mid_i64 for @Mid[i64] where [@X[i64]] { trait.assoc_type @Out = i64 }

trait.trait private @Wrap[!S] { trait.assoc_type @Item }
trait.impl private @Wrap_i64 for @Wrap[i64] { trait.assoc_type @Item = !trait.proj<@Mid[i64], "Out"> }

trait.trait private @Run[!S] {
  func.func private @go(!S) -> i64
}

func.func private @need(%v: i64, %e: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64 {
  return %v : i64
}

trait.impl private @Run_gen for @Run[!S] where [@Wrap[!S], !trait.proj<@Wrap[!S], "Item"> = i64] {
  func.func @go(%x: !S) -> i64 {
    %e = trait.assume !trait.proj<@Wrap[!S], "Item"> = i64
    %v = arith.constant 7 : i64
    %r = trait.func.call @need(%v, %e)
      : (i64, !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64
    return %r : i64
  }
}

// CHECK: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@Run[i64]>'
func.func @main() -> i64 {
  %c = trait.allege @Run[i64]
  %x = arith.constant 3 : i64
  %r = trait.method.call %c @Run[i64]::@go(%x) : (i64) -> i64
  return %r : i64
}
