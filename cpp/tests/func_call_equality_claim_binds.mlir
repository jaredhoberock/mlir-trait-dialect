// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// A generic callee whose formal is an equality claim over a type variable is
// callable across a boundary: instantiating the call unifies the formal's
// endpoints against the actual's, so claim<!poly<0> = i32> against
// claim<i64 = i32> binds !poly<0> := i64 and the callee's !poly<0> result is the
// caller's i64. An equality claim's endpoints live in storage the generic
// instantiation replacer cannot see, so before they were instantiated across the
// boundary the rigid variable rejected every such call.

!S = !trait.poly<0>

func.func @gen(%c: !trait.claim<!S = i32>) -> !S {
  %x = arith.constant 0 : i32
  %v = trait.coerce %x : i32 to !S via (%c) : (!trait.claim<!S = i32>)
  return %v : !S
}

// CHECK-LABEL: func.func @caller
// CHECK: trait.func.call @gen(%{{.*}}) : (!trait.claim<i64 = i32>) -> i64
func.func @caller(%e: !trait.claim<i64 = i32>) -> i64 {
  %r = trait.func.call @gen(%e) : (!trait.claim<i64 = i32>) -> i64
  return %r : i64
}
