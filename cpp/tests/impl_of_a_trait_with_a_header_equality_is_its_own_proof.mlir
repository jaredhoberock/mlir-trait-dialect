// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @T's header states an equality and requires no application, and @T_i64 binds
// Out so that the equality holds. Nothing stands between the impl and the
// claim, so impl selection records the impl itself as the proof of @T[i64] and
// every reader of that citation reads it by the same predicate: a citation
// naming an impl carries no subproofs, so what it may not meet is an
// application requirement.

trait.trait private @T[!trait.poly<0>] where [!trait.proj<@T[!trait.poly<0>], "Out"> = i64] {
  trait.assoc_type @Out
  func.func private @get(!trait.poly<0>) -> !trait.proj<@T[!trait.poly<0>], "Out">
}
trait.impl private @T_i64 for @T[i64] {
  trait.assoc_type @Out = i64
  func.func @get(%x: i64) -> i64 {
    %c = arith.constant 5 : i64
    return %c : i64
  }
}

// CHECK-NOT: trait.
// CHECK: func.func private @T_i64_get(%[[X:.*]]: i64) -> i64
// CHECK: func.func @main(%[[A:.*]]: i64) -> i64
// CHECK: %[[R:.*]] = call @T_i64_get(%[[A]]) : (i64) -> i64
// CHECK: return %[[R]] : i64
// CHECK-NOT: trait.
func.func @main(%x: i64) -> i64 {
  %c = trait.allege @T[i64]
  %r = trait.method.call %c @T[i64]::@get(%x) : (i64) -> !trait.proj<@T[i64], "Out">
  %o = trait.coerce %r : !trait.proj<@T[i64], "Out"> to i64 unproven
  return %o : i64
}
