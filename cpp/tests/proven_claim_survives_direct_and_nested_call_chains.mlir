// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// The @P[i32] proof is used by both a direct generic call and a nested call
// that also needs @Q[i32]. Proof propagation must preserve both concrete call
// chains: @user1 to @g, and @user2 to @h to @g2.

trait.trait private @P[!trait.poly<0>] {}
trait.impl private @P_impl for @P[!trait.poly<1>] {}

trait.trait private @Q[!trait.poly<2>] {}
trait.impl private @Q_impl for @Q[!trait.poly<3>] {}

!T = !trait.poly<4>
func.func private @g(%c: !trait.claim<@P[!T]>, %x: !T) -> !T {
  return %x : !T
}

!U = !trait.poly<5>
func.func private @g2(%q: !trait.claim<@Q[!U]>, %c: !trait.claim<@P[!U]>, %x: !U) -> !U {
  return %x : !U
}

!V = !trait.poly<6>
func.func private @h(%c: !trait.claim<@P[!V]>, %x: !V) -> !V {
  %q = trait.derive @Q[!V] from @Q_impl given()
  %r = trait.func.call @g2(%q, %c, %x)
    : (!trait.claim<@Q[!V]>, !trait.claim<@P[!V]>, !V) -> !V
  return %r : !V
}

func.func @user1(%x: i32) -> i32 {
  %p = trait.allege @P[i32]
  %r = trait.func.call @g(%p, %x) : (!trait.claim<@P[i32]>, i32) -> i32
  return %r : i32
}

func.func @user2(%x: i32) -> i32 {
  %p = trait.allege @P[i32]
  %r = trait.func.call @h(%p, %x) : (!trait.claim<@P[i32]>, i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func private @g_h
// CHECK: return %arg0 : i32
// CHECK-LABEL: func.func private @g2_h
// CHECK: return %arg0 : i32
// CHECK-LABEL: func.func private @h_h
// CHECK: call @g2_h
// CHECK-LABEL: func.func @user1
// CHECK: call @g_h
// CHECK-LABEL: func.func @user2
// CHECK: call @h_h
