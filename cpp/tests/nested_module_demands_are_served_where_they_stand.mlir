// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// A demand is put to selection in the module that stands around the op
// spelling it, and served by that module's impls. Here a nested module
// declares @Has and the root does not, and the projection a parameter of the
// nested module's @g spells resolves through the nested module's impl.

// CHECK: module @inner
// CHECK: func.func @g(%{{.*}}: i64) -> i64
func.func @main() -> i64 {
  %c = arith.constant 0 : i64
  return %c : i64
}
module @inner {
  trait.trait private @Has(!trait.claim<@Has[!trait.poly<0>]>) assoc {Out = 0}
  trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) bindings {Out = i64} {
  }
  func.func @g(%x: !trait.proj<@Has[i32], "Out">) -> i64 {
    %c = arith.constant 0 : i64
    return %c : i64
  }
}

// -----

// The evidence an impl of a nested module returns for its trait's requirement
// is an allegation, which the instance projecting it leaves to selection. The
// demand that leaves is answered in the nested module, where the projection
// reads it, so the call through the projected requirement runs @A_inner.

// CHECK: module @inner
// CHECK: func.func private @[[A:A_inner.a_h[0-9a-f]+]]() -> i64
// CHECK: func.func private @[[B:B_impl.b_h[0-9a-f]+]]() -> i64
// CHECK:   call @[[A]]()
// CHECK: func.func @main() -> i64
// CHECK:   call @[[B]]()
func.func @outer() { return }
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
    %req0 = trait.allege @A[i32]
    trait.return %req0 : !trait.claim<@A[i32]>
  }
  func.func private @B_impl.b(%self: !trait.claim<@B[i32]>) -> i64 {
    %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
    %r = trait.func.call @A.a(%a) : (!trait.claim<@A[i32]>) -> i64
    return %r : i64
  }
  func.func @main() -> i64 {
    %c = trait.allege @B[i32]
    %r = trait.func.call @B.b(%c) : (!trait.claim<@B[i32]>) -> i64
    return %r : i64
  }
}
