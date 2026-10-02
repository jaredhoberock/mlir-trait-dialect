// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The control of invalid_false_equality_an_evidence_method_returns_is_decided:
// @eq computes evidence for i64 = i64, the allegation the call is replaced by
// holds, and the program runs.

// CHECK: {{^}}9{{$}}

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
 trait.method @eq() -> !trait.claim<i64 = i64>
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
 trait.method @eq() -> !trait.claim<i64 = i64> {
  %e = trait.allege i64 = i64
  trait.return %e : !trait.claim<i64 = i64>
 }
}
func.func @main() -> i64 {
 %a = trait.witness @I for @A[i8]
 %e = trait.method.call %a @A[i8]::@eq() : () -> !trait.claim<i64 = i64> by @I
 %n = arith.constant 9 : i64
 %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<i64 = i64>)
 return %r : i64
}
