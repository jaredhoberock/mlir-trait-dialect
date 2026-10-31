// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The refusal guard on the two-hop settlement. The chain's first hop grounds
// (@Wrap_i64 binds Item to @Mid[i64]::Out), but the second hop's impl @Mid_i64
// is conditional on @X[i64], for which the module carries no impl. A projection
// resolves only through an impl whose obligations hold, so @Mid[i64]::Out stays
// spelled and the premise's endpoints never meet at a ground type. Discharging
// the premise is what selection does, so @Run_gen is no candidate for the
// demand and the allegation has no impl -- the equality is never settled on the
// strength of @Mid_i64's binding alone. Supplying an @X[i64] impl is what
// discharges the where entry and lets the premise hold.

// VERIFY: trait.impl private @Run_gen({{.*}}, %item: !trait.claim<!trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64>)
// VERIFY: func.func private @Run_gen.go({{.*}}, %[[ITEM:[^:]*]]: !trait.claim<!trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64>, %{{[^:]*}}: !trait.poly<0>)
// VERIFY: trait.func.call @need(%{{.*}}, %[[ITEM]])

!S = !trait.poly<0>

trait.trait private @X(!trait.claim<@X[!trait.poly<0>]>)
trait.trait private @Mid(!trait.claim<@Mid[!S]>) assoc {Out = 0}
trait.impl private @Mid_i64(%self: !trait.claim<@Mid[i64]>, %x: !trait.claim<@X[i64]>) bindings {Out = i64} {
}

trait.trait private @Wrap(!trait.claim<@Wrap[!S]>) assoc {Item = 0}
trait.impl private @Wrap_i64(%self: !trait.claim<@Wrap[i64]>) bindings {Item = !trait.proj<@Mid[i64], "Out">} {
}

trait.trait private @Run(!trait.claim<@Run[!S]>) members [@Run.go]
func.func private @Run.go(!trait.claim<@Run[!S]>, !S) -> i64

func.func private @need(%v: i64, %e: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64 {
  return %v : i64
}

trait.impl private @Run_gen(%self: !trait.claim<@Run[!S]>, %wrap: !trait.claim<@Wrap[!S]>, %item: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) members [@Run.go -> @Run_gen.go] {
}
func.func private @Run_gen.go(%self: !trait.claim<@Run[!S]>, %wrap: !trait.claim<@Wrap[!S]>, %item: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>, %x: !S) -> i64 {
  %v = arith.constant 7 : i64
  %r = trait.func.call @need(%v, %item)
    : (i64, !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64
  return %r : i64
}

// CHECK: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@Run[i64]>'
func.func @main() -> i64 {
  %c = trait.allege @Run[i64]
  %x = arith.constant 3 : i64
  %r = trait.func.call @Run.go(%c, %x) : (!trait.claim<@Run[i64]>, i64) -> i64
  return %r : i64
}
