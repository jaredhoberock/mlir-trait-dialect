// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One call supplies @T1[i64] for two reasons: through @T2's requirement, which
// @T2_i64 returns as @T1_i64_a (7), and as @f's own claim parameter,
// carried by @T1_i64_b (9). No respelling keyed by @T1[i64] can say which
// proof a spelling of it means. The parameter takes the evidence supplied at
// its position, and the projection of @T2's requirement takes its source's
// requirement at its index, so each member call runs the impl its own evidence
// selects.

// CHECK: {{^}}16{{$}}

trait.trait private @T0(!trait.claim<@T0[!trait.poly<0>]>) assoc {A = 0}

trait.trait private @T1(!trait.claim<@T1[!trait.poly<0>]>) members [@T1.value]
func.func private @T1.value(!trait.claim<@T1[!trait.poly<0>]>) -> i64

trait.trait private @T2(!trait.claim<@T2[!trait.poly<0>]>) -> !trait.claim<@T1[!trait.proj<@T0[!trait.poly<0>], "A">]>

trait.impl private @T1_i64_a(%self: !trait.claim<@T1[i64]>) members [@T1.value -> @T1_i64_a.value] {
}
func.func private @T1_i64_a.value(%self: !trait.claim<@T1[i64]>) -> i64 {
  %v = arith.constant 7 : i64
  return %v : i64
}
trait.impl private @T1_i64_b(%self: !trait.claim<@T1[i64]>) members [@T1.value -> @T1_i64_b.value] {
}
func.func private @T1_i64_b.value(%self: !trait.claim<@T1[i64]>) -> i64 {
  %v = arith.constant 9 : i64
  return %v : i64
}

trait.impl private @T0_i64(%self: !trait.claim<@T0[i64]>) bindings {A = i64} {
}

trait.impl private @T2_i64(%self: !trait.claim<@T2[i64]>) {
  %t1 = trait.witness @T1_i64_a for @T1[i64]
  %eq = trait.witness proj_resolve !trait.proj<@T0[i64], "A"> resolves i64 by @T0_i64
    : !trait.claim<!trait.proj<@T0[i64], "A"> = i64>
  %req0 = trait.coerce %t1
    : !trait.claim<@T1[i64] by @T1_i64_a>
    to !trait.claim<@T1[!trait.proj<@T0[i64], "A">]>
    via (%eq) : (!trait.claim<!trait.proj<@T0[i64], "A"> = i64>)
  trait.return %req0 : !trait.claim<@T1[!trait.proj<@T0[i64], "A">]>
}

trait.proof private @T2_i64_p {
  %d = trait.derive @T2[i64] from @T2_i64 given()
  trait.return %d : !trait.claim<@T2[i64]>
}

func.func private @f(
  %x: !trait.poly<0>,
  %t2: !trait.claim<@T2[!trait.poly<0>]>,
  %t1: !trait.claim<@T1[!trait.proj<@T0[!trait.poly<0>], "A">]>
) -> i64 {
  %r = trait.project %t2[0]
    : !trait.claim<@T2[!trait.poly<0>]>
    -> !trait.claim<@T1[!trait.proj<@T0[!trait.poly<0>], "A">]>
  %u = trait.func.call @T1.value(%r) : (!trait.claim<@T1[!trait.proj<@T0[!trait.poly<0>], "A">]>) -> i64
  %v = trait.func.call @T1.value(%t1) : (!trait.claim<@T1[!trait.proj<@T0[!trait.poly<0>], "A">]>) -> i64
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
    to !trait.claim<@T1[!trait.proj<@T0[i64], "A">]>
    via (%eq) : (!trait.claim<!trait.proj<@T0[i64], "A"> = i64>)
  %r = trait.func.call @f(%x, %t2, %t1_projected)
    : (i64,
       !trait.claim<@T2[i64] by @T2_i64_p>,
       !trait.claim<@T1[!trait.proj<@T0[i64], "A">]>)
    -> i64
  return %r : i64
}
