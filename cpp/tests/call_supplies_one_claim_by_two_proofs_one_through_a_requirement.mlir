// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One call supplies @T1[i64] for two reasons: through @T2's requirement, which
// @T2_i64_p discharges with @T1_i64_a (7), and as @f's own claim parameter,
// carried by @T1_i64_b (9). No respelling keyed by @T1[i64] can say which
// proof a spelling of it means. The parameter takes the evidence supplied at
// its position, and the projection of @T2's requirement takes its source's
// requirement at its index, so each method call runs the impl its own evidence
// selects.

// CHECK: {{^}}16{{$}}

trait.trait private @T0[!trait.poly<0>] {
  trait.assoc_type @A
}

trait.trait private @T1[!trait.poly<1>] { trait.method @value() -> i64 }

trait.trait private @T2[!trait.poly<2>] where [@T1[!trait.proj<@T0[!trait.poly<2>], "A">]] {}

trait.impl private @T1_i64_a for @T1[i64] {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @T1_i64_b for @T1[i64] {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}

trait.impl private @T0_i64 for @T0[i64] {
  trait.assoc_type @A = i64
}

trait.impl private @T2_i64 for @T2[i64] {}

trait.proof private @T2_i64_p proves @T2_i64[] for @T2[i64] given [@T1_i64_a]

func.func private @f(
  %x: !trait.poly<3>,
  %t2: !trait.claim<@T2[!trait.poly<3>]>,
  %t1: !trait.claim<@T1[!trait.proj<@T0[!trait.poly<3>], "A">]>
) -> i64 {
  %r = trait.project %t2[0]
    : !trait.claim<@T2[!trait.poly<3>]>
    -> !trait.claim<@T1[!trait.proj<@T0[!trait.poly<3>], "A">]>
  %u = trait.method.call %r @T1[!trait.proj<@T0[!trait.poly<3>], "A">]::@value() : () -> i64
  %v = trait.method.call %t1 @T1[!trait.proj<@T0[!trait.poly<3>], "A">]::@value() : () -> i64
  %s = arith.addi %u, %v : i64
  return %s : i64
}

func.func @main() -> i64 {
  %x = arith.constant 1 : i64
  %t2 = trait.witness @T2_i64_p for @T2[i64]
  %t1 = trait.witness @T1_i64_b for @T1[i64]
  %eq = trait.witness proj_resolve !trait.proj<@T0[i64], "A"> resolves i64 by @T0_i64
    : !trait.claim<!trait.proj<@T0[i64], "A"> = i64>
  %t1_projected = trait.coerce %t1
    : !trait.claim<@T1[i64] by @T1_i64_b>
    to !trait.claim<@T1[!trait.proj<@T0[i64], "A">] by @T1_i64_b>
    via (%eq) : (!trait.claim<!trait.proj<@T0[i64], "A"> = i64>)
  %r = trait.func.call @f(%x, %t2, %t1_projected)
    : (i64,
       !trait.claim<@T2[i64] by @T2_i64_p>,
       !trait.claim<@T1[!trait.proj<@T0[i64], "A">] by @T1_i64_b>)
    -> i64
  return %r : i64
}
