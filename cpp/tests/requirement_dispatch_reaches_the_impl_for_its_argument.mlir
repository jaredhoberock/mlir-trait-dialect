// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The counterpart of the refused citation: with nothing standing over @B[i32],
// selection discharges its requirement @A[i32] with @A_i32, and the call
// @B_i32's body makes through that requirement reaches @A_i32's member.

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>) -> i64
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> members [@B.b]
func.func private @B.b(!trait.claim<@B[!trait.poly<0>]>, !trait.poly<0>) -> i64
trait.impl private @A_i32(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_i32.a] {
}
func.func private @A_i32.a(%self: !trait.claim<@A[i32]>) -> i64 {
  %c = arith.constant 32 : i64
  return %c : i64
}
trait.impl private @A_i64(%self: !trait.claim<@A[i64]>) members [@A.a -> @A_i64.a] {
}
func.func private @A_i64.a(%self: !trait.claim<@A[i64]>) -> i64 {
  %c = arith.constant 64 : i64
  return %c : i64
}
trait.impl private @B_i32(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_i32.b] {
  %req0 = trait.allege @A[i32]
  trait.return %req0 : !trait.claim<@A[i32]>
}
func.func private @B_i32.b(%self: !trait.claim<@B[i32]>, %x: i32) -> i64 {
  %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
  %r = trait.func.call @A.a(%a) : (!trait.claim<@A[i32]>) -> i64
  return %r : i64
}

// CHECK: func.func private @[[A32:A_i32.a_h[0-9a-f]+]]() -> i64
// CHECK: arith.constant 32
// CHECK: func.func private @[[B:B_i32.b_h[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: call @[[A32]]
// CHECK: func.func @main
// CHECK: call @[[B]]
func.func @main(%x: i32) -> i64 {
  %c = trait.allege @B[i32]
  %r = trait.func.call @B.b(%c, %x) : (!trait.claim<@B[i32]>, i32) -> i64
  return %r : i64
}
