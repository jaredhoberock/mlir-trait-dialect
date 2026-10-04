// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: not mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait)' 2>&1 | FileCheck %s

// @I's method @eq computes evidence for i32 = i64 by an allegation, and main
// cites what a call of it returns in an identity coerce. The coerce settles at
// once, and the call, which computes evidence, stands although nothing reads
// it: it is replaced by @eq's body, so the allegation stands where the call
// stood, an obligation outstanding, and is decided there, and refused. The
// true control is equality_an_evidence_method_returns_is_decided_where_it_holds.
// CHECK: :[[@LINE+8]]:{{[0-9]+}}: error: 'trait.allege' op alleges 'i32' = 'i64', and impl selection resolves its sides to 'i32' and 'i64'

!T = !trait.poly<0>
trait.trait private @A(%self: !trait.claim<@A[!T]>) {
 trait.method @eq() -> !trait.claim<i32 = i64>
}
trait.impl private @I(%self: !trait.claim<@A[i8]>) {
 trait.method @eq() -> !trait.claim<i32 = i64> {
  %e = trait.allege i32 = i64
  trait.return %e : !trait.claim<i32 = i64>
 }
}
func.func @main() -> i64 {
 %a = trait.witness @I for @A[i8]
 %e = trait.method.call %a @A[i8]::@eq() : () -> !trait.claim<i32 = i64> by @I
 %n = arith.constant 9 : i64
 %r = trait.coerce %n : i64 to i64 via (%e) : (!trait.claim<i32 = i64>)
 return %r : i64
}
