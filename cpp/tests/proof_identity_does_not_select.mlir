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
// The trait's member is cloned once per impl for the same reason, so the two
// i32 callers share a member clone as well as a callee clone.

!S = !trait.poly<0>

trait.trait private @U(!trait.claim<@U[!S]>)

trait.trait private @T(!trait.claim<@T[!S]>) assoc {Out = 0} members [@T.m]
func.func private @T.m(!trait.claim<@T[!S]>, !S) -> !trait.proj<@T[!S], "Out">

trait.impl private @U_i32(%self: !trait.claim<@U[i32]>) {
}

trait.impl private @T_cond(%self: !trait.claim<@T[!S]>, %u: !trait.claim<@U[!S]>) bindings {Out = i64} members [@T.m -> @T_cond.m] {
}
func.func private @T_cond.m(%self: !trait.claim<@T[!S]>, %u: !trait.claim<@U[!S]>, %x: !S) -> i64 {
  %c = arith.constant 7 : i64
  return %c : i64
}

trait.impl private @T_f64(%self: !trait.claim<@T[f64]>) bindings {Out = i1} members [@T.m -> @T_f64.m] {
}
func.func private @T_f64.m(%self: !trait.claim<@T[f64]>, %x: f64) -> i1 {
  %c = arith.constant true
  return %c : i1
}

trait.proof private @P {
  %p0 = trait.witness @U_i32 for @U[i32]
  %d = trait.derive @T[i32] from @T_cond[i32] given(%p0) : (!trait.claim<@U[i32] by @U_i32>)
  trait.return %d : !trait.claim<@T[i32]>
}

func.func private @g(%c: !trait.claim<@T[!S]>, %x: !S) -> !trait.proj<@T[!S], "Out"> {
  %r = trait.func.call @T.m(%c, %x) : (!trait.claim<@T[!S]>, !S) -> !trait.proj<@T[!S], "Out">
  return %r : !trait.proj<@T[!S], "Out">
}

// CHECK: func.func private @[[MC:T_cond.m_h[0-9a-f]+]](%{{.*}}: i32) -> i64
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
  %r = trait.func.call @g(%e, %x)
    : (!trait.claim<@T[i32] by @P>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @through_an_allegation(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.allege @T[i32]
  %r = trait.func.call @g(%e, %x)
    : (!trait.claim<@T[i32]>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @at_the_other_type(%x: f64) -> !trait.proj<@T[f64], "Out"> {
  %e = trait.allege @T[f64]
  %r = trait.func.call @g(%e, %x)
    : (!trait.claim<@T[f64]>, f64) -> !trait.proj<@T[f64], "Out">
  return %r : !trait.proj<@T[f64], "Out">
}
