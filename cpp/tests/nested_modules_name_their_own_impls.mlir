// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Two modules spell one application and mean two different impls of it. What
// impl selection settles is settled for the module the demand was read in, so
// the inner module's demand reaches the impl standing there and names a symbol
// its own symbol table resolves -- not the one the module around it holds under
// another name.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) {
  trait.method @m() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}
// CHECK: func.func private @T_i32_{{h[0-9a-f]+}}_m
// CHECK: arith.constant 1
// CHECK: func.func @main
// CHECK: call @T_i32_{{h[0-9a-f]+}}_m
func.func @main() -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.method.call %c @T[i32]::@m() : () -> i64
  return %r : i64
}

// CHECK: module @inner
module @inner {
  trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
  trait.impl private @T_inner(%self: !trait.claim<@T[i32]>) {
    trait.method @m() -> i64 {
      %c = arith.constant 2 : i64
      trait.return %c : i64
    }
  }
  // CHECK: func.func private @T_inner_{{h[0-9a-f]+}}_m
  // CHECK: arith.constant 2
  // CHECK: func.func @main
  // CHECK: call @T_inner_{{h[0-9a-f]+}}_m
  func.func @main() -> i64 {
    %c = trait.allege @T[i32]
    %r = trait.method.call %c @T[i32]::@m() : () -> i64
    return %r : i64
  }
}

// -----

// The same rule for a proof already written. The outer module holds a proof of
// @B_impl at @B[i32], whose @B_impl returns @A_top for its requirement; the
// inner module spells the trait and the impl the same way, its @B_impl returns
// @A_inner, and it has no proof of its own. A proof stands over the impl it
// names in the module it is written in, so the outer proof is no answer for the
// inner demand, which selection serves with the inner @B_impl, and its method
// reaches @A_inner.

trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> { trait.method @b() -> i64 }
trait.impl private @A_top(%self: !trait.claim<@A[i32]>) {
  trait.method @a() -> i64 {
    %c = arith.constant 1 : i64
    trait.return %c : i64
  }
}
trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) {
  %top = trait.witness @A_top for @A[i32]
  trait.method @b() -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
    %r = trait.method.call %a @A[i32]::@a() : () -> i64
    trait.return %r : i64
  }
  trait.return %top : !trait.claim<@A[i32] by @A_top>
}
trait.proof private @p {
  %d = trait.derive @B[i32] from @B_impl given()
  trait.return %d : !trait.claim<@B[i32]>
}
// CHECK: func.func private @A_top_{{h[0-9a-f]+}}_a
// CHECK: func.func private @B_impl_{{h[0-9a-f]+}}_b
// CHECK: call @A_top_{{h[0-9a-f]+}}_a
func.func @main() -> i64 {
  %w = trait.witness @p for @B[i32]
  %r = trait.method.call %w @B[i32]::@b() : () -> i64 by @p
  return %r : i64
}

// CHECK: module @inner
module @inner {
  trait.trait private @A(%self: !trait.claim<@A[!trait.poly<0>]>) { trait.method @a() -> i64 }
  trait.trait private @B(%self: !trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> { trait.method @b() -> i64 }
  trait.impl private @A_inner(%self: !trait.claim<@A[i32]>) {
    trait.method @a() -> i64 {
      %c = arith.constant 2 : i64
      trait.return %c : i64
    }
  }
  trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) {
    %inner = trait.witness @A_inner for @A[i32]
    trait.method @b() -> i64 {
      %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
      %r = trait.method.call %a @A[i32]::@a() : () -> i64
      trait.return %r : i64
    }
    trait.return %inner : !trait.claim<@A[i32] by @A_inner>
  }
  // CHECK: func.func private @A_inner_{{h[0-9a-f]+}}_a
  // CHECK: func.func private @B_impl_{{h[0-9a-f]+}}_b
  // CHECK: call @A_inner_{{h[0-9a-f]+}}_a
  func.func @main() -> i64 {
    %c = trait.allege @B[i32]
    %r = trait.method.call %c @B[i32]::@b() : () -> i64
    return %r : i64
  }
}
