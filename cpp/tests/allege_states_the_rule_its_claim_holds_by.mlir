// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s --check-prefix=UNSERVED

// An allegation names the rule its claim holds by and the facts the rule
// consulted. The identity is the implementing dialect's and opaque here: the
// claim is proved by asking generation for that rule, and a rule no loaded
// generator implements proves nothing.

// CHECK: trait.allege @A[i32] by "some.rule" given(%{{.*}} : !trait.claim<@B[i32]>)
// CHECK: trait.allege @A[!trait.poly<0>] by "some.rule" unsafe
// UNSERVED: error: 'trait.allege' op no impl with satisfiable assumptions for '!trait.claim<@A[i32]>'

trait.trait private @A[!trait.poly<0>] {
  func.func private @a() -> i64
}
trait.trait private @B[!trait.poly<0>] {}
trait.impl private @B_i32 for @B[i32] {}

func.func @main() -> i64 {
  %b = trait.allege @B[i32]
  %a = trait.allege @A[i32] by "some.rule" given(%b : !trait.claim<@B[i32]>)
  %r = trait.method.call %a @A[i32]::@a() : () -> i64
  return %r : i64
}

func.func private @f() {
  %a = trait.allege @A[!trait.poly<0>] by "some.rule" unsafe
  return
}
