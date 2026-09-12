// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Two modules spell one application and mean two different impls of it. What
// impl selection settles is settled for the module the demand was read in, so
// the inner module's demand reaches the impl standing there and names a symbol
// its own symbol table resolves -- not the one the module around it holds under
// another name.

trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
trait.impl private @T_i32 for @T[i32] {
  func.func @m() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
// CHECK: func.func private @T_i32_m
// CHECK: arith.constant 1
// CHECK: func.func @main
// CHECK: call @T_i32_m
func.func @main() -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.method.call %c @T[i32]::@m() : () -> i64
  return %r : i64
}

// CHECK: module @inner
module @inner {
  trait.trait private @T[!trait.poly<0>] { func.func private @m() -> i64 }
  trait.impl private @T_inner for @T[i32] {
    func.func @m() -> i64 {
      %c = arith.constant 2 : i64
      return %c : i64
    }
  }
  // CHECK: func.func private @T_inner_m
  // CHECK: arith.constant 2
  // CHECK: func.func @main
  // CHECK: call @T_inner_m
  func.func @main() -> i64 {
    %c = trait.allege @T[i32]
    %r = trait.method.call %c @T[i32]::@m() : () -> i64
    return %r : i64
  }
}

// -----

// The same rule for a proof already written. The outer module holds a proof of
// @B_impl at @B[i32] discharging its requirement with @A_top; the inner module
// spells the trait and the impl the same way and has @A_inner and no proof of
// its own. A proof stands over the impl it names in the module it is written
// in, so the outer proof is no answer for the inner demand, which builds the
// proof citing @A_inner.

trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] { func.func private @b() -> i64 }
trait.impl private @A_top for @A[i32] {
  func.func @a() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
trait.impl private @B_impl for @B[i32] {
  func.func @b() -> i64 {
    %s = trait.assume @B[i32]
    %a = trait.project %s : @B[i32] to @A[i32]
    %r = trait.method.call %a @A[i32]::@a() : () -> i64
    return %r : i64
  }
}
trait.proof private @p proves @B_impl for @B[i32] given [@A_top]
// CHECK: func.func private @A_top_a
// CHECK: func.func private @B_impl_b
// CHECK: call @A_top_a
func.func @main() -> i64 {
  %w = trait.witness @p for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @p
  return %r : i64
}

// CHECK: module @inner
module @inner {
  trait.trait private @A[!trait.poly<0>] { func.func private @a() -> i64 }
  trait.trait private @B[!trait.poly<0>] where [@A[!trait.poly<0>]] { func.func private @b() -> i64 }
  trait.impl private @A_inner for @A[i32] {
    func.func @a() -> i64 {
      %c = arith.constant 2 : i64
      return %c : i64
    }
  }
  trait.impl private @B_impl for @B[i32] {
    func.func @b() -> i64 {
      %s = trait.assume @B[i32]
      %a = trait.project %s : @B[i32] to @A[i32]
      %r = trait.method.call %a @A[i32]::@a() : () -> i64
      return %r : i64
    }
  }
  // CHECK: func.func private @A_inner_a
  // CHECK: func.func private @B_impl_b
  // CHECK: call @A_inner_a
  func.func @main() -> i64 {
    %c = trait.allege @B[i32]
    %r = trait.method.call %c @B[i32]::@b() : () -> i64
    return %r : i64
  }
}
