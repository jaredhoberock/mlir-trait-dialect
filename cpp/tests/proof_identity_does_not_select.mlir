// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// Which clone a call reaches is decided by the claim's type, never by the
// evidence that discharged it. @T_cond is conditional and is reached through a
// premise; @T_f64 is unconditional and is reached directly. A caller carrying
// the proof, a caller alleging and letting selection find it, and a caller at
// the other type all pass through one generic, and what comes out is one clone
// per type -- the mangled name reads the specialization alone.
//
// The trait's method is cloned once per impl for the same reason, so the two
// i32 callers share a method clone as well as a callee clone.

!S = !trait.poly<0>

trait.trait private @U[!S] {
}

trait.trait private @T[!S] {
  trait.assoc_type @Out
  func.func private @m(!S) -> !trait.proj<@T[!S], "Out">
}

trait.impl private @U_i32 for @U[i32] {
}

trait.impl private @T_cond for @T[!S] where [@U[!S]] {
  trait.assoc_type @Out = i64
  func.func private @m(%x: !S) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

trait.impl private @T_f64 for @T[f64] {
  trait.assoc_type @Out = i1
  func.func private @m(%x: f64) -> i1 {
    %c = arith.constant true
    return %c : i1
  }
}

trait.proof private @P proves @T_cond for @T[i32] given [@U_i32]

func.func private @g(%c: !trait.claim<@T[!S]>, %x: !S) -> !trait.proj<@T[!S], "Out"> {
  %r = trait.method.call %c @T[!S]::@m(%x) : (!S) -> !trait.proj<@T[!S], "Out">
  return %r : !trait.proj<@T[!S], "Out">
}

// CHECK: func.func private @[[MC:T_cond_[0-9a-z_]+]](%{{.*}}: i32) -> i64
// CHECK: func.func private @[[G32:g_h[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: call @[[MC]]
// CHECK: func.func private @[[G64:g_h[0-9a-f]+]](%{{.*}}: f64) -> i1
// CHECK: func.func @through_the_proof
// CHECK: call @[[G32]]
// CHECK: func.func @through_an_allegation
// CHECK: call @[[G32]]
// CHECK: func.func @at_the_other_type
// CHECK: call @[[G64]]
func.func @through_the_proof(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.witness @P for @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32] by @P>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @through_an_allegation(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.allege @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32]>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @at_the_other_type(%x: f64) -> !trait.proj<@T[f64], "Out"> {
  %e = trait.allege @T[f64]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [f64]}
    : (!trait.claim<@T[f64]>, f64) -> !trait.proj<@T[f64], "Out">
  return %r : !trait.proj<@T[f64], "Out">
}
