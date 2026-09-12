// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The counterpart of the refused citation: with nothing standing over @B[i32],
// selection discharges its requirement @A[i32] with @A_i32, and the call
// @B_i32's body makes through that requirement reaches @A_i32's method.

trait.trait private @A[!trait.poly<0>] {
  func.func private @a() -> i64
}
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {
  func.func private @b(!trait.poly<0>) -> i64
}
trait.impl private @A_i32 for @A[i32] {
  func.func @a() -> i64 {
    %c = arith.constant 32 : i64
    return %c : i64
  }
}
trait.impl private @A_i64 for @A[i64] {
  func.func @a() -> i64 {
    %c = arith.constant 64 : i64
    return %c : i64
  }
}
trait.impl private @B_i32 for @B[i32] {
  func.func @b(%x: i32) -> i64 {
    %s = trait.assume @B[i32]
    %a = trait.project %s : @B[i32] to @A[i32]
    %r = trait.method.call %a @A[i32]::@a() : () -> i64
    return %r : i64
  }
}

// CHECK: func.func private @[[A32:A_i32_a]]() -> i64
// CHECK: arith.constant 32
// CHECK: func.func private @[[B:B_i32_b]](%{{.*}}: i32) -> i64
// CHECK: call @[[A32]]
// CHECK: func.func @main
// CHECK: call @[[B]]
func.func @main(%x: i32) -> i64 {
  %c = trait.allege @B[i32]
  %r = trait.method.call %c @B[i32]::@b(%x) : (i32) -> i64
  return %r : i64
}
