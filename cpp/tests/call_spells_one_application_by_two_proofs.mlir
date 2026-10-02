// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// @pv1 and @pv2 both stand over @T_i64 at @T[i64]. Each names a declaration
// that carries to the claim it is spelled on, which is the whole of what a
// spelling asserts, so a call carrying one application under both symbols
// verifies: they are two names for one fact, and which name a clone takes is
// settled where the clone is cut.

trait.trait private @T(%self: !trait.claim<@T[!trait.poly<0>]>) { trait.method @t() -> i64 }
trait.impl private @T_i64(%self: !trait.claim<@T[i64]>) {
  trait.method @t() -> i64 {
    %c = arith.constant 5 : i64
    trait.return %c : i64
  }
}
trait.proof private @pv1 {
  %d = trait.derive @T[i64] from @T_i64 given()
  trait.return %d : !trait.claim<@T[i64]>
}
trait.proof private @pv2 {
  %d = trait.derive @T[i64] from @T_i64 given()
  trait.return %d : !trait.claim<@T[i64]>
}

func.func private @callee(!trait.claim<@T[i64]>, !trait.claim<@T[i64]>)

// CHECK-LABEL: func.func @two_names
// CHECK: trait.func.call @callee(%{{.*}}, %{{.*}}) : (!trait.claim<@T[i64] by @pv1>, !trait.claim<@T[i64] by @pv2>) -> ()
func.func @two_names() {
  %a = trait.witness @pv1 for @T[i64]
  %b = trait.witness @pv2 for @T[i64]
  trait.func.call @callee(%a, %b)
    : (!trait.claim<@T[i64] by @pv1>, !trait.claim<@T[i64] by @pv2>) -> ()
  return
}
