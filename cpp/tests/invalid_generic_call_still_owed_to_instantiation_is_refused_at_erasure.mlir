// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(erase-polymorphs-trait)' -verify-diagnostics
// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A method call whose receiver arrives through two coerces, the first dropping
// the proof, is a generic call instantiation lowers: the module is pending, and
// the lowering driver runs instantiation first, after which the call is the
// impl's method and the program returns 32. Handed to erasure alone, the call
// still stands, and erasure's conversion would erase the evidence it reads
// under it; so erasure's entry refuses the work instantiation still owes, at
// the call, before anything else runs.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {
  trait.method @m() -> i64 { %v = arith.constant 32 : i64  trait.return %v : i64 }
}

// CHECK-LABEL: func.func @f() -> i64
// CHECK: call @{{.*}}() : () -> i64
func.func @f() -> i64 {
  %w = trait.witness @T_i32 for @T[i32]
  %c1 = trait.coerce %w : !trait.claim<@T[i32] by @T_i32> to !trait.claim<@T[i32]>
  %c2 = trait.coerce %c1 : !trait.claim<@T[i32]> to !trait.claim<@T[i32]>
  // expected-error @below {{'trait.method.call' op is a generic call instantiate-monomorphs has not lowered before erasure}}
  %v = trait.method.call %c2 @T[i32]::@m() : () -> i64
  return %v : i64
}
