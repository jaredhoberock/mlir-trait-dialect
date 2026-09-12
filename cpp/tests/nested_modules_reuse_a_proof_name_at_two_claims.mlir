// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// The counterpart: one proof name in two modules standing over two different
// claims. Each module's @p proves its own application, and the method each call
// reaches is its own module's.

trait.trait private @A[!trait.poly<0>] {}
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {
  func.func private @m(!trait.poly<0>) -> !trait.poly<0>
}
trait.impl private @A_top for @A[i32] {}
trait.impl private @B_impl for @B[i32] {
  func.func @m(%x: i32) -> i32 { return %x : i32 }
}
trait.proof private @p proves @B_impl for @B[i32] given [@A_top]

// CHECK: func.func private @B_impl_m(%{{.*}}: i32) -> i32
// CHECK: func.func @main(%{{.*}}: i32) -> i32
func.func @main(%x: i32) -> i32 {
  %w = trait.witness @p for @B[i32]
  %r = trait.method.call %w @B[i32]::@m(%x) : (i32) -> i32 by @p
  return %r : i32
}

// CHECK: module @inner
module @inner {
  trait.trait private @A[!trait.poly<0>] {}
  trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] {
    func.func private @m(!trait.poly<0>) -> !trait.poly<0>
  }
  trait.impl private @A_inner for @A[i64] {}
  trait.impl private @B_impl for @B[i64] {
    func.func @m(%x: i64) -> i64 { return %x : i64 }
  }
  trait.proof private @p proves @B_impl for @B[i64] given [@A_inner]

  // CHECK: func.func private @B_impl_m(%{{.*}}: i64) -> i64
  // CHECK: func.func @main(%{{.*}}: i64) -> i64
  func.func @main(%x: i64) -> i64 {
    %w = trait.witness @p for @B[i64]
    %r = trait.method.call %w @B[i64]::@m(%x) : (i64) -> i64 by @p
    return %r : i64
  }
}
