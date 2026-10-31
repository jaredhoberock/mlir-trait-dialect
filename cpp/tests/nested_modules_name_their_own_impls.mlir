// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Two modules spell one application and mean two different impls of it. What
// impl selection settles is settled for the module the demand was read in, so
// the inner module's demand reaches the impl standing there and names a symbol
// its own symbol table resolves -- not the one the module around it holds under
// another name.

trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.m]
func.func private @T.m(!trait.claim<@T[!trait.poly<0>]>) -> i64
trait.impl private @T_i32(%self: !trait.claim<@T[i32]>) members [@T.m -> @T_i32.m] {
}
func.func private @T_i32.m(%self: !trait.claim<@T[i32]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
// CHECK: func.func private @T_i32.m_{{h[0-9a-f]+}}
// CHECK: arith.constant 1
// CHECK: func.func @main
// CHECK: call @T_i32.m_{{h[0-9a-f]+}}
func.func @main() -> i64 {
  %c = trait.allege @T[i32]
  %r = trait.func.call @T.m(%c) : (!trait.claim<@T[i32]>) -> i64
  return %r : i64
}

// CHECK: module @inner
module @inner {
  trait.trait private @T(!trait.claim<@T[!trait.poly<0>]>) members [@T.m]
  func.func private @T.m(!trait.claim<@T[!trait.poly<0>]>) -> i64
  trait.impl private @T_inner(%self: !trait.claim<@T[i32]>) members [@T.m -> @T_inner.m] {
  }
  func.func private @T_inner.m(%self: !trait.claim<@T[i32]>) -> i64 {
    %c = arith.constant 2 : i64
    return %c : i64
  }
  // CHECK: func.func private @T_inner.m_{{h[0-9a-f]+}}
  // CHECK: arith.constant 2
  // CHECK: func.func @main
  // CHECK: call @T_inner.m_{{h[0-9a-f]+}}
  func.func @main() -> i64 {
    %c = trait.allege @T[i32]
    %r = trait.func.call @T.m(%c) : (!trait.claim<@T[i32]>) -> i64
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

trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>) -> i64
trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> members [@B.b]
func.func private @B.b(!trait.claim<@B[!trait.poly<0>]>) -> i64
trait.impl private @A_top(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_top.a] {
}
func.func private @A_top.a(%self: !trait.claim<@A[i32]>) -> i64 {
  %c = arith.constant 1 : i64
  return %c : i64
}
trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_impl.b] {
  %top = trait.witness @A_top for @A[i32]
  trait.return %top : !trait.claim<@A[i32] by @A_top>
}
func.func private @B_impl.b(%self: !trait.claim<@B[i32]>) -> i64 {
  %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
  %r = trait.func.call @A.a(%a) : (!trait.claim<@A[i32]>) -> i64
  return %r : i64
}
trait.proof private @p {
  %d = trait.derive @B[i32] from @B_impl given()
  trait.return %d : !trait.claim<@B[i32]>
}
// CHECK: func.func private @A_top.a_{{h[0-9a-f]+}}
// CHECK: func.func private @B_impl.b_{{h[0-9a-f]+}}
// CHECK: call @A_top.a_{{h[0-9a-f]+}}
func.func @main() -> i64 {
  %w = trait.witness @p for @B[i32]
  %r = trait.func.call @B.b(%w) : (!trait.claim<@B[i32] by @p>) -> i64
  return %r : i64
}

// CHECK: module @inner
module @inner {
  trait.trait private @A(!trait.claim<@A[!trait.poly<0>]>) members [@A.a]
  func.func private @A.a(!trait.claim<@A[!trait.poly<0>]>) -> i64
  trait.trait private @B(!trait.claim<@B[!trait.poly<0>]>) -> !trait.claim<@A[!trait.poly<0>]> members [@B.b]
  func.func private @B.b(!trait.claim<@B[!trait.poly<0>]>) -> i64
  trait.impl private @A_inner(%self: !trait.claim<@A[i32]>) members [@A.a -> @A_inner.a] {
  }
  func.func private @A_inner.a(%self: !trait.claim<@A[i32]>) -> i64 {
    %c = arith.constant 2 : i64
    return %c : i64
  }
  trait.impl private @B_impl(%self: !trait.claim<@B[i32]>) members [@B.b -> @B_impl.b] {
    %inner = trait.witness @A_inner for @A[i32]
    trait.return %inner : !trait.claim<@A[i32] by @A_inner>
  }
  func.func private @B_impl.b(%self: !trait.claim<@B[i32]>) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
    %r = trait.func.call @A.a(%a) : (!trait.claim<@A[i32]>) -> i64
    return %r : i64
  }
  // CHECK: func.func private @A_inner.a_{{h[0-9a-f]+}}
  // CHECK: func.func private @B_impl.b_{{h[0-9a-f]+}}
  // CHECK: call @A_inner.a_{{h[0-9a-f]+}}
  func.func @main() -> i64 {
    %c = trait.allege @B[i32]
    %r = trait.func.call @B.b(%c) : (!trait.claim<@B[i32]>) -> i64
    return %r : i64
  }
}
