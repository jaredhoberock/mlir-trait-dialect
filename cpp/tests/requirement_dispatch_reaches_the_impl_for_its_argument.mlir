// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The counterpart of the refused citation: with nothing standing over @B[i32],
// selection discharges its requirement @A[i32] with @A_i32, and the call
// @B_i32's body makes through that requirement reaches @A_i32's method.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {
  trait.method @a() -> i64
}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  trait.method @b(!trait.poly<0>) -> i64
}
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 32 : i64
    trait.return %c : i64
  }
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 64 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) {
  trait.method @b(%x: i32) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
    %r = trait.method.call %a @A[i32]::@a() : () -> i64
    trait.return %r : i64
  }
  %req0 = trait.allege @A[i32]
  trait.return %req0 : !trait.claim<@A[i32]>
}

// CHECK: func.func private @[[A32:A_i32_h[0-9a-f]+_a]]() -> i64
// CHECK: arith.constant 32
// CHECK: func.func private @[[B:B_i32_h[0-9a-f]+_b]](%{{.*}}: i32) -> i64
// CHECK: call @[[A32]]
// CHECK: func.func @main
// CHECK: call @[[B]]
func.func @main(%x: i32) -> i64 {
  %c = trait.allege @B[i32]
  %r = trait.method.call %c @B[i32]::@b(%x) : (i32) -> i64
  return %r : i64
}
