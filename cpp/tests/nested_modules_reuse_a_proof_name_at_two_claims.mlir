// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The counterpart: one proof name in two modules standing over two different
// claims. Each module's @p proves its own application, and the method each call
// reaches is its own module's.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
  trait.method @m(!trait.poly<0>) -> !trait.poly<0>
}
trait.impl private @A_top(%self: !trait.claim<@A[i32]>) {}
trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) {
  trait.method @m(%x: i32) -> i32 { trait.return %x : i32 }
  %req0 = trait.allege @A[i32]
  trait.return %req0 : !trait.claim<@A[i32]>
}
trait.proof private @p {
  %d = trait.derive @B[i32] from @B_impl given()
  trait.return %d : !trait.claim<@B[i32]>
}

// CHECK: func.func private @B_impl_{{h[0-9a-f]+}}_m(%{{.*}}: i32) -> i32
// CHECK: func.func @main(%{{.*}}: i32) -> i32
func.func @main(%x: i32) -> i32 {
  %w = trait.witness @p for @B[i32]
  %r = trait.method.call %w @B[i32]::@m(%x) : (i32) -> i32 by @p
  return %r : i32
}

// CHECK: module @inner
module @inner {
  trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) {}
  trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> {
    trait.method @m(!trait.poly<0>) -> !trait.poly<0>
  }
  trait.impl private @A_inner(%self: !trait.claim<@A[i64]>) {}
  trait.impl private @B_impl(%self: !trait.claim<@B[i64]>) {
    trait.method @m(%x: i64) -> i64 { trait.return %x : i64 }
  %req0 = trait.allege @A[i64]
  trait.return %req0 : !trait.claim<@A[i64]>
}
  trait.proof private @p {
    %d = trait.derive @B[i64] from @B_impl given()
    trait.return %d : !trait.claim<@B[i64]>
  }

  // CHECK: func.func private @B_impl_{{h[0-9a-f]+}}_m(%{{.*}}: i64) -> i64
  // CHECK: func.func @main(%{{.*}}: i64) -> i64
  func.func @main(%x: i64) -> i64 {
    %w = trait.witness @p for @B[i64]
    %r = trait.method.call %w @B[i64]::@m(%x) : (i64) -> i64 by @p
    return %r : i64
  }
}
