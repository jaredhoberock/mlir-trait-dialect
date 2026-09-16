// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A generic callee takes @U[T] and @V[T]. The caller passes @U[i8] by @pu,
// whose subproof for @V[i8] is the blanket @pv, and @V[i8] by @pv. The same
// claim reaches the call twice: once through the first operand's proof tree
// and once as an operand of its own, and a subproof's claim is the obligation
// it discharges at the application the citation names, so both spell it at i8.
// Split 2 swaps the operand order.

trait.trait private @V[!trait.poly<0>] { func.func private @v() -> i64 }
trait.trait private @U[!trait.poly<0>] where [@V[!trait.poly<0>]] { func.func private @u() -> i64 }
trait.impl private @V_blanket for @V[!trait.poly<0>] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @U_blanket for @U[!trait.poly<0>] {
  func.func @u() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
trait.proof private @pv proves @V_blanket for @V[!trait.poly<0>] given []
trait.proof private @pu proves @U_blanket for @U[!trait.poly<0>] given [@pv]

// CHECK-NOT: trait.
// CHECK: func.func private @[[V:V_blanket_[a-z0-9]+]]_v() -> i64
// CHECK: func.func private @[[F:f_[a-z0-9]+]]() -> i64
// CHECK: call @[[V]]_v() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @[[F]]() : () -> i64
// CHECK-NOT: trait.
func.func private @f(%u: !trait.claim<@U[!trait.poly<0>]>, %v: !trait.claim<@V[!trait.poly<0>]>) -> i64 {
  %r = trait.method.call %v @V[!trait.poly<0>]::@v() : () -> i64
  return %r : i64
}
func.func @main() -> i64 {
  %wu = trait.witness @pu for @U[i8]
  %wv = trait.witness @pv for @V[i8]
  %r = trait.func.call @f(%wu, %wv) : (!trait.claim<@U[i8] by @pu>, !trait.claim<@V[i8] by @pv>) -> i64
  return %r : i64
}

// -----

trait.trait private @V[!trait.poly<0>] { func.func private @v() -> i64 }
trait.trait private @U[!trait.poly<0>] where [@V[!trait.poly<0>]] { func.func private @u() -> i64 }
trait.impl private @V_blanket for @V[!trait.poly<0>] {
  func.func @v() -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}
trait.impl private @U_blanket for @U[!trait.poly<0>] {
  func.func @u() -> i64 {
    %c = arith.constant 1 : i64
    return %c : i64
  }
}
trait.proof private @pv proves @V_blanket for @V[!trait.poly<0>] given []
trait.proof private @pu proves @U_blanket for @U[!trait.poly<0>] given [@pv]

// CHECK-NOT: trait.
// CHECK: func.func private @[[V2:V_blanket_[a-z0-9]+]]_v() -> i64
// CHECK: func.func private @[[F2:f_[a-z0-9]+]]() -> i64
// CHECK: call @[[V2]]_v() : () -> i64
// CHECK: func.func @main() -> i64
// CHECK: call @[[F2]]() : () -> i64
// CHECK-NOT: trait.
func.func private @f(%v: !trait.claim<@V[!trait.poly<0>]>, %u: !trait.claim<@U[!trait.poly<0>]>) -> i64 {
  %r = trait.method.call %v @V[!trait.poly<0>]::@v() : () -> i64
  return %r : i64
}
func.func @main() -> i64 {
  %wu = trait.witness @pu for @U[i8]
  %wv = trait.witness @pv for @V[i8]
  %r = trait.func.call @f(%wv, %wu) : (!trait.claim<@V[i8] by @pv>, !trait.claim<@U[i8] by @pu>) -> i64
  return %r : i64
}
