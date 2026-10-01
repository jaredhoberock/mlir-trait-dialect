// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(instantiate-monomorphs-trait)' | FileCheck %s --check-prefix=INSTANCES
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// One callee, one type argument, two calls whose claim operands name different
// proofs of @T[i64], @pv1 and @pv2. An instance is named by the evidence it is
// made with, never by its claim type alone, so the calls reach two instances,
// each taking the proof its call supplied; both run @T_i64's method.

// INSTANCES-DAG: func.func private @[[G1:g_h[0-9a-f]+]](%{{.*}}: !trait.claim<@T[i64] by @pv1>) -> i64
// INSTANCES-DAG: func.func private @[[G2:g_h[0-9a-f]+]](%{{.*}}: !trait.claim<@T[i64] by @pv2>) -> i64
// INSTANCES-DAG: call @[[G1]](%{{.*}}) : (!trait.claim<@T[i64] by @pv1>) -> i64
// INSTANCES-DAG: call @[[G2]](%{{.*}}) : (!trait.claim<@T[i64] by @pv2>) -> i64

// CHECK: {{^}}10{{$}}

trait.trait private @T[!trait.poly<0>] { func.func private @t() -> i64 }
trait.impl private @T_i64 for @T[i64] {
  func.func @t() -> i64 {
    %c = arith.constant 5 : i64
    return %c : i64
  }
}
trait.proof private @pv1 proves @T_i64[] for @T[i64] given []
trait.proof private @pv2 proves @T_i64[] for @T[i64] given []

func.func private @g(%c: !trait.claim<@T[!trait.poly<3>]>) -> i64 {
  %r = trait.method.call %c @T[!trait.poly<3>]::@t() : () -> i64
  return %r : i64
}

func.func @main() -> i64 {
  %w1 = trait.witness @pv1 for @T[i64]
  %w2 = trait.witness @pv2 for @T[i64]
  %a = trait.func.call @g(%w1)
    : (!trait.claim<@T[i64] by @pv1>) -> i64
  %b = trait.func.call @g(%w2)
    : (!trait.claim<@T[i64] by @pv2>) -> i64
  %s = arith.addi %a, %b : i64
  return %s : i64
}
