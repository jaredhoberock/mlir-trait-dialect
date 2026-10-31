// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s --check-prefix=VERIFY
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The impl's where-clause equality <@Wrap[i64]::Item = i64>, which its member
// function takes as its leading parameter after the impl's own, is forwarded to
// a callee operand whose monomorphic instance retains the equality-claim
// parameter. The member call passes the equality premise of the proof selection
// mints, cloned from the proof's body, to the member function's instance. The equality holds only across two resolution
// hops: @Wrap_i64 binds Item to @Mid[i64]::Out, and @Mid_i64 binds that Out to
// i64. Selection grounds both hops -- a single hop would leave @Mid[i64]::Out
// still spelled and the equality unproven -- and erasure removes the evidence.

// VERIFY: trait.impl private @Run_gen({{.*}}, %item: !trait.claim<!trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64>)
// VERIFY: func.func private @Run_gen.go({{.*}}, %[[ITEM:[^:]*]]: !trait.claim<!trait.proj<@Wrap[!trait.poly<0>], "Item"> = i64>, %{{[^:]*}}: !trait.poly<0>)
// VERIFY: trait.func.call @need(%{{.*}}, %[[ITEM]])

!S = !trait.poly<0>

trait.trait private @Mid(!trait.claim<@Mid[!S]>) assoc {Out = 0}
trait.impl private @Mid_i64(%self: !trait.claim<@Mid[i64]>) bindings {Out = i64} {
}

trait.trait private @Wrap(!trait.claim<@Wrap[!S]>) assoc {Item = 0}
trait.impl private @Wrap_i64(%self: !trait.claim<@Wrap[i64]>) bindings {Item = !trait.proj<@Mid[i64], "Out">} {
}

trait.trait private @Run(!trait.claim<@Run[!S]>) members [@Run.go]
func.func private @Run.go(!trait.claim<@Run[!S]>, !S) -> i64

// The callee's monomorphic instance keeps the equality-claim parameter on its
// ABI even though the body ignores it: the parameter carries the evidence
// across the call boundary.
func.func private @need(%v: i64, %e: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64 {
  return %v : i64
}

// The closure-like impl: its where-clause carries the inherited equality; the
// member function reads it as its leading parameter and forwards it as the call
// operand.
trait.impl private @Run_gen(%self: !trait.claim<@Run[!S]>, %wrap: !trait.claim<@Wrap[!S]>, %item: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) members [@Run.go -> @Run_gen.go] {
}
func.func private @Run_gen.go(%self: !trait.claim<@Run[!S]>, %wrap: !trait.claim<@Wrap[!S]>, %item: !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>, %x: !S) -> i64 {
  %v = arith.constant 7 : i64
  %r = trait.func.call @need(%v, %item)
    : (i64, !trait.claim<!trait.proj<@Wrap[!S], "Item"> = i64>) -> i64
  return %r : i64
}

func.func @main() -> i64 {
  %c = trait.allege @Run[i64]
  %x = arith.constant 3 : i64
  %r = trait.func.call @Run.go(%c, %x) : (!trait.claim<@Run[i64]>, i64) -> i64
  return %r : i64
}

// The two-hop chain settles clean: no trait op survives.
// CHECK: func.func private @need
// CHECK-NOT: trait.
// CHECK: func.func @main
