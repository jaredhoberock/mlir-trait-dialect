// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: timeout 120 mlir-opt %s -split-input-file -pass-pipeline='builtin.module(monomorphize-trait)' | FileCheck %s

// @J's header projects Item of @A[i32], whose binding grows without end, so
// the header has no normal form and @J is no candidate for @B[i64]; @K serves
// it, and the member call runs @K's member function. The growth stands beneath
// a tuple here and at the binding's root below: both are rounds of one
// reading, which meets one bound and hands its caller the same non-answer
// either way.

// CHECK-LABEL: func.func @main
// CHECK: call @K.value_{{.*}}
!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = tuple<!trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>>} {
}
trait.trait private @B(!trait.claim<@B[!T]>) members [@B.value]
func.func private @B.value(!trait.claim<@B[!T]>) -> i64
trait.impl private @J(%s: !trait.claim<@B[!trait.proj<@A[i32], "Item", [i64]>]>) members [@B.value -> @J.value] {
}
func.func private @J.value(%s: !trait.claim<@B[!trait.proj<@A[i32], "Item", [i64]>]>) -> i64 {
  %c = arith.constant 1 : i64 return %c : i64
}
trait.impl private @K(%s: !trait.claim<@B[i64]>) members [@B.value -> @K.value] {
}
func.func private @K.value(%s: !trait.claim<@B[i64]>) -> i64 {
  %c = arith.constant 7 : i64 return %c : i64
}
func.func @main() -> i64 {
  %b = trait.allege @B[i64]
  %v = trait.func.call @B.value(%b) : (!trait.claim<@B[i64]>) -> i64
  return %v : i64
}

// -----

// CHECK-LABEL: func.func @main
// CHECK: call @K.value_{{.*}}
!T = !trait.poly<0>
trait.trait private @A(!trait.claim<@A[!T]>) assoc {Item = 1}
trait.impl private @I(%s: !trait.claim<@A[i32]>) bindings {Item = !trait.proj<@A[i32], "Item", [tuple<!trait.poly<0>>]>} {
}
trait.trait private @B(!trait.claim<@B[!T]>) members [@B.value]
func.func private @B.value(!trait.claim<@B[!T]>) -> i64
trait.impl private @J(%s: !trait.claim<@B[!trait.proj<@A[i32], "Item", [i64]>]>) members [@B.value -> @J.value] {
}
func.func private @J.value(%s: !trait.claim<@B[!trait.proj<@A[i32], "Item", [i64]>]>) -> i64 {
  %c = arith.constant 1 : i64 return %c : i64
}
trait.impl private @K(%s: !trait.claim<@B[i64]>) members [@B.value -> @K.value] {
}
func.func private @K.value(%s: !trait.claim<@B[i64]>) -> i64 {
  %c = arith.constant 7 : i64 return %c : i64
}
func.func @main() -> i64 {
  %b = trait.allege @B[i64]
  %v = trait.func.call @B.value(%b) : (!trait.claim<@B[i64]>) -> i64
  return %v : i64
}
