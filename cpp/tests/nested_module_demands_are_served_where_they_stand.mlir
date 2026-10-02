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
  trait.trait private @Has(%self: !trait.claim<@Has[!trait.poly<0>]>) { trait.assoc_type @Out }
  trait.impl private @Has_i32(%self: !trait.claim<@Has[i32]>) { trait.assoc_type @Out = i64 }
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
// CHECK: func.func private @[[A:A_inner_h[0-9a-f]+]]_a() -> i64
// CHECK: func.func private @[[B:B_impl_h[0-9a-f]+]]_b() -> i64
// CHECK:   call @[[A]]_a()
// CHECK: func.func @main() -> i64
// CHECK:   call @[[B]]_b()
func.func @outer() { return }
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
    trait.method @b() -> i64 {
      %a = trait.project %self[0] : !trait.claim<@B[i32]> -> !trait.claim<@A[i32]>
      %r = trait.method.call %a @A[i32]::@a() : () -> i64
      trait.return %r : i64
    }
    %req0 = trait.allege @A[i32]
    trait.return %req0 : !trait.claim<@A[i32]>
  }
  func.func @main() -> i64 {
    %c = trait.allege @B[i32]
    %r = trait.method.call %c @B[i32]::@b() : () -> i64
    return %r : i64
  }
}
