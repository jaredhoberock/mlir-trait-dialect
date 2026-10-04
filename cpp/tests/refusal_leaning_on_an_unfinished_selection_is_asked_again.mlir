// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// Selecting for @A[i32] reads its one candidate's premise @C[i32], whose
// candidate @C_via_b reads @B[i32], whose one candidate reads @A[i32] again.
// The cycle guard refuses that, so @B[i32] is refused while @A[i32] is still
// being selected; @C_direct then serves @C[i32], and @A[i32] is selected after
// all. The refusal of @B[i32] leaned on a selection that was not finished and
// finished otherwise, so it is no answer: asked on its own, @B[i32] is served
// by @B_via_a, and its method runs.

// CHECK: {{^}}7{{$}}

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
  trait.method @value() -> i64
}
trait.trait private @B(%self: !trait.claim<@B[!T]>) {
  trait.method @value() -> i64
}
trait.trait private @C(%self: !trait.claim<@C[!T]>) {}
trait.trait private @D(%self: !trait.claim<@D[!T]>) {}
trait.impl private @D_i32(%self: !trait.claim<@D[i32]>) {}

trait.impl private @A_via_c(%self: !trait.claim<@A[!T]>, %c: !trait.claim<@C[!T]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 3 : i64
    trait.return %v : i64
  }
}
trait.impl private @C_via_b(%self: !trait.claim<@C[!T]>, %b: !trait.claim<@B[!T]>) {}
trait.impl private @C_direct(%self: !trait.claim<@C[i32]>, %d: !trait.claim<@D[i32]>) {}
trait.impl private @B_via_a(%self: !trait.claim<@B[!T]>, %a: !trait.claim<@A[!T]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 4 : i64
    trait.return %v : i64
  }
}

// The stage reaches a function's last op first, so @A[i32] is asked before
// @B[i32].
func.func @main() -> i64 {
  %b = trait.allege @B[i32]
  %vb = trait.method.call %b @B[i32]::@value() : () -> i64
  %a = trait.allege @A[i32]
  %va = trait.method.call %a @A[i32]::@value() : () -> i64
  %v = arith.addi %va, %vb : i64
  return %v : i64
}
