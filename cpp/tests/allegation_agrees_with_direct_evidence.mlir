// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// An allegation names no impl and a witness names one, and the two must reach
// the same program. Selection follows the claim's type: both callers below land
// on one clone of the generic callee, with the projection in its result resolved
// the same way. The modules that follow put the same pair of callers under a
// permuted impl order, which adds a second instantiation and no second answer
// for the first, and under a differently named impl, whose name the callers
// carry but selection does not read.

!S = !trait.poly<0>

trait.trait private @T[!S] {
  trait.assoc_type @Out
  func.func private @m(!S) -> !trait.proj<@T[!S], "Out">
}

trait.impl private @T_i32 for @T[i32] {
  trait.assoc_type @Out = i64
  func.func private @m(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

func.func private @g(%c: !trait.claim<@T[!S]>, %x: !S) -> !trait.proj<@T[!S], "Out"> {
  %r = trait.method.call %c @T[!S]::@m(%x) : (!S) -> !trait.proj<@T[!S], "Out">
  return %r : !trait.proj<@T[!S], "Out">
}

// CHECK: func.func private @g_h[[H:[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: func.func @by_witness
// CHECK: call @g_h[[H]]
// CHECK: func.func @by_allegation
// CHECK: call @g_h[[H]]
func.func @by_witness(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.witness @T_i32 for @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32] by @T_i32>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @by_allegation(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.allege @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32]>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

// -----

// A second impl standing first in the module. Selection reads the claim's type,
// so the i32 callers still meet at one clone and the f64 caller gets its own.

!S = !trait.poly<0>

trait.trait private @T[!S] {
  trait.assoc_type @Out
  func.func private @m(!S) -> !trait.proj<@T[!S], "Out">
}

trait.impl private @T_f64 for @T[f64] {
  trait.assoc_type @Out = i1
  func.func private @m(%x: f64) -> i1 {
    %c = arith.constant true
    return %c : i1
  }
}

trait.impl private @T_i32 for @T[i32] {
  trait.assoc_type @Out = i64
  func.func private @m(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

func.func private @g(%c: !trait.claim<@T[!S]>, %x: !S) -> !trait.proj<@T[!S], "Out"> {
  %r = trait.method.call %c @T[!S]::@m(%x) : (!S) -> !trait.proj<@T[!S], "Out">
  return %r : !trait.proj<@T[!S], "Out">
}

// CHECK: func.func private @g_h[[G32:[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: func.func private @g_h[[G64:[0-9a-f]+]](%{{.*}}: f64) -> i1
// CHECK: func.func @permuted_by_witness
// CHECK: call @g_h[[G32]]
// CHECK: func.func @permuted_by_allegation
// CHECK: call @g_h[[G32]]
// CHECK: func.func @selects_by_type
// CHECK: call @g_h[[G64]]
func.func @permuted_by_witness(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.witness @T_i32 for @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32] by @T_i32>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @permuted_by_allegation(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.allege @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32]>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @selects_by_type(%x: f64) -> !trait.proj<@T[f64], "Out"> {
  %e = trait.allege @T[f64]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [f64]}
    : (!trait.claim<@T[f64]>, f64) -> !trait.proj<@T[f64], "Out">
  return %r : !trait.proj<@T[f64], "Out">
}

// -----

// The same program under a differently named impl. The witness carries the name
// and the allegation carries none, and both reach the one clone.

!S = !trait.poly<0>

trait.trait private @T[!S] {
  trait.assoc_type @Out
  func.func private @m(!S) -> !trait.proj<@T[!S], "Out">
}

trait.impl private @Chosen for @T[i32] {
  trait.assoc_type @Out = i64
  func.func private @m(%x: i32) -> i64 {
    %c = arith.constant 7 : i64
    return %c : i64
  }
}

func.func private @g(%c: !trait.claim<@T[!S]>, %x: !S) -> !trait.proj<@T[!S], "Out"> {
  %r = trait.method.call %c @T[!S]::@m(%x) : (!S) -> !trait.proj<@T[!S], "Out">
  return %r : !trait.proj<@T[!S], "Out">
}

// CHECK: func.func private @g_h[[R:[0-9a-f]+]](%{{.*}}: i32) -> i64
// CHECK: func.func @renamed_by_witness
// CHECK: call @g_h[[R]]
// CHECK: func.func @renamed_by_allegation
// CHECK: call @g_h[[R]]
func.func @renamed_by_witness(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.witness @Chosen for @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32] by @Chosen>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}

func.func @renamed_by_allegation(%x: i32) -> !trait.proj<@T[i32], "Out"> {
  %e = trait.allege @T[i32]
  %r = trait.func.call @g(%e, %x) {type_params = [!trait.poly<0>], type_args = [i32]}
    : (!trait.claim<@T[i32]>, i32) -> !trait.proj<@T[i32], "Out">
  return %r : !trait.proj<@T[i32], "Out">
}
